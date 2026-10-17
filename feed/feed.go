// Package feed is the EmberGate fork feed (hearth W-051, 2026-10-17): the model server's own
// record of what went IN to the weights and what came OUT, written by the process that owns
// them, with no harbor code in the path.
//
// Per /api/chat request it writes, as JSON lines:
//
//	open   the EXACT rendered prompt string after the Go template (what is tokenized), the
//	       model name + blob digest, the options as applied, counts, and the harbor turn id
//	       (echo of the X-Harbor-Turn request header) so the two feeds join by id.
//	piece  every piece the runner emitted, in order, BEFORE ollama parses it into
//	       thinking / content / tool_calls; with draft (MTP) counts when the runner sends them.
//	out    every response line as sent to the client, byte for byte (the tee sits after
//	       json.Marshal, before the write), so harbor's chunk feed can be checked against it.
//	close  how the turn ended: done reason, eval / prompt counts, errors, pieces, outs, drops.
//
// File: <OLLAMA_FEED_DIR>/<utc date>/ollama-feed-<utc date>.jsonl, append, one event per line.
// Never blocks generation: events go through a buffered channel to one writer goroutine; when
// the buffer is full the event is dropped and counted (the close event carries the total).
// Enabled by OLLAMA_FEED_DIR; unset = every call is a no-op. The feed holds the kid's text
// verbatim (drawer included) like harbor's captures: masking is a READ-side job.
package feed

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"time"
)

// Schema names the event shape; bump it when a field changes meaning.
const Schema = "ollama-feed.v1"

// ContextKey is where the handler parks the current turn on the gin context so the
// generic stream writer can tee the bytes it sends (see OutFrom).
const ContextKey = "embergate.feed.turn"

// EnvDir is the environment variable that enables the feed and names its directory.
const EnvDir = "OLLAMA_FEED_DIR"

const queueSize = 1 << 16

// Feed is one writer; Default is built from the environment at init.
type Feed struct {
	dir     string
	ch      chan []byte
	dropped atomic.Int64
	program   map[string]any
	once      sync.Once
	stampOnce sync.Once
}

// Default is the process-wide feed (nil-safe: a disabled feed is a nil *Feed).
var Default = FromEnv()

// FromEnv returns the feed named by OLLAMA_FEED_DIR, or nil when unset.
func FromEnv() *Feed {
	dir := os.Getenv(EnvDir)
	if dir == "" {
		return nil
	}
	return New(dir)
}

// New makes a feed writing under dir. The writer goroutine starts on first use.
func New(dir string) *Feed {
	return &Feed{dir: dir, ch: make(chan []byte, queueSize)}
}

// Enabled reports whether events are recorded.
func (f *Feed) Enabled() bool { return f != nil }

// Dropped is the number of events lost to backpressure since start.
func (f *Feed) Dropped() int64 {
	if f == nil {
		return 0
	}
	return f.dropped.Load()
}

func programStamp() map[string]any {
	p := map[string]any{"version": Version, "vcs_revision": "unknown", "vcs_modified": "unknown"}
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			switch s.Key {
			case "vcs.revision":
				p["vcs_revision"] = s.Value
			case "vcs.modified":
				p["vcs_modified"] = s.Value
			}
		}
	}
	return p
}

// Version is set by the server package from ollama's version package (routes.go) — after this
// package's init, which is why the program stamp is computed lazily on first Open.
var Version = "unknown"

func newID() string {
	var b [6]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "feed" + hex.EncodeToString([]byte(time.Now().Format("150405.000")))[:8]
	}
	return hex.EncodeToString(b[:])
}

func now() float64 { return float64(time.Now().UnixNano()) / 1e9 }

func (f *Feed) emit(ev map[string]any) {
	if f == nil {
		return
	}
	f.once.Do(func() { go f.run() })
	ev["schema"] = Schema
	if _, ok := ev["ts"]; !ok {
		ev["ts"] = now()
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false) // least processed: '<think>' stays '<think>', not <think>
	if err := enc.Encode(ev); err != nil {
		f.dropped.Add(1)
		return
	}
	select {
	case f.ch <- buf.Bytes():
	default:
		f.dropped.Add(1)
	}
}

func (f *Feed) run() {
	var fh *os.File
	var day, lastErr string
	fail := func(err error) {
		if lastErr != err.Error() {
			slog.Error("embergate feed write failed; events dropped until it recovers", "error", err, "dir", f.dir)
			lastErr = err.Error()
		}
		if fh != nil {
			fh.Close()
			fh = nil
		}
	}
	for b := range f.ch {
		d := time.Now().UTC().Format("2006-01-02")
		if fh == nil || d != day {
			if fh != nil {
				fh.Close()
				fh = nil
			}
			dir := filepath.Join(f.dir, d)
			if err := os.MkdirAll(dir, 0o755); err != nil {
				fail(err)
				continue
			}
			h, err := os.OpenFile(filepath.Join(dir, "ollama-feed-"+d+".jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
			if err != nil {
				fail(err)
				continue
			}
			fh, day = h, d
		}
		if _, err := fh.Write(b); err != nil {
			fail(err)
			continue
		}
		if lastErr != "" {
			slog.Info("embergate feed writing again", "dir", f.dir)
			lastErr = ""
		}
	}
}

// Turn is one /api/chat request's record. All methods are nil-safe.
type Turn struct {
	f      *Feed
	ID     string // the feed's own id for this turn
	Harbor string // X-Harbor-Turn as received ("" when the caller sent none)
	start  time.Time
	pieces atomic.Int64
	outs   atomic.Int64

	mu       sync.Mutex
	final    *Piece
	errs     []string
	finished bool
}

// Piece is what the runner emitted in one callback, before any ollama-side parsing.
type Piece struct {
	Content               string
	Done                  bool
	DoneReason            string
	EvalCount             int
	PromptEvalCount       int
	PromptEvalCachedCount *int
	DraftN                int
	DraftAcceptedN        int
}

// Open records the turn's input side and returns its Turn (nil when the feed is disabled).
// fields carries what the handler knows: rendered_prompt, model, options, counts.
func (f *Feed) Open(harborTurn string, fields map[string]any) *Turn {
	if f == nil {
		return nil
	}
	t := &Turn{f: f, ID: newID(), Harbor: harborTurn, start: time.Now()}
	f.stampOnce.Do(func() { f.program = programStamp() })
	ev := map[string]any{"ev": "open", "turn": harborTurn, "feed_turn": t.ID, "program": f.program}
	for k, v := range fields {
		ev[k] = v
	}
	f.emit(ev)
	return t
}

// Piece records one runner callback.
func (t *Turn) Piece(p Piece) {
	if t == nil {
		return
	}
	seq := t.pieces.Add(1)
	ev := map[string]any{"ev": "piece", "turn": t.Harbor, "feed_turn": t.ID, "seq": seq, "content": p.Content}
	if p.DraftN != 0 || p.DraftAcceptedN != 0 {
		ev["draft_n"] = p.DraftN
		ev["draft_n_accepted"] = p.DraftAcceptedN
	}
	if p.Done {
		ev["done"] = true
		ev["done_reason"] = p.DoneReason
		t.mu.Lock()
		pc := p
		t.final = &pc
		t.mu.Unlock()
	}
	t.f.emit(ev)
}

// Out records one response line exactly as written to the client.
func (t *Turn) Out(raw []byte) {
	if t == nil {
		return
	}
	seq := t.outs.Add(1)
	t.f.emit(map[string]any{"ev": "out", "turn": t.Harbor, "feed_turn": t.ID, "seq": seq, "raw": string(raw)})
}

// Finish notes how generation ended (errors are kept for the close event; nil is fine).
func (t *Turn) Finish(errs ...error) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.finished = true
	for _, e := range errs {
		if e != nil {
			t.errs = append(t.errs, e.Error())
		}
	}
}

// Close writes the close event. Call it once, after the response has been fully written.
func (t *Turn) Close() {
	if t == nil {
		return
	}
	t.mu.Lock()
	ev := map[string]any{
		"ev": "close", "turn": t.Harbor, "feed_turn": t.ID,
		"pieces": t.pieces.Load(), "outs": t.outs.Load(),
		"dropped_total": t.f.Dropped(), "duration_ms": time.Since(t.start).Milliseconds(),
		"finished": t.finished, "errors": t.errs,
	}
	if t.final != nil {
		ev["done_reason"] = t.final.DoneReason
		ev["eval_count"] = t.final.EvalCount
		ev["prompt_eval_count"] = t.final.PromptEvalCount
		ev["prompt_eval_cached_count"] = t.final.PromptEvalCachedCount
		ev["draft_n"] = t.final.DraftN
		ev["draft_n_accepted"] = t.final.DraftAcceptedN
	} else {
		ev["done_reason"] = nil
	}
	t.mu.Unlock()
	t.f.emit(ev)
}

// Getter is the slice of gin.Context the stream writer needs.
type Getter interface {
	Get(key string) (any, bool)
}

// OutFrom tees raw to the turn parked on the context, if any. Zero cost when the feed is off.
func OutFrom(c Getter, raw []byte) {
	if c == nil {
		return
	}
	if v, ok := c.Get(ContextKey); ok {
		if t, ok := v.(*Turn); ok {
			t.Out(raw)
		}
	}
}
