package feed

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func readEvents(t *testing.T, dir string) []map[string]any {
	t.Helper()
	var out []map[string]any
	matches, _ := filepath.Glob(filepath.Join(dir, "*", "ollama-feed-*.jsonl"))
	for _, p := range matches {
		fh, err := os.Open(p)
		if err != nil {
			t.Fatal(err)
		}
		sc := bufio.NewScanner(fh)
		sc.Buffer(make([]byte, 1<<20), 1<<24)
		for sc.Scan() {
			var ev map[string]any
			if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
				t.Fatalf("bad line %q: %v", sc.Text(), err)
			}
			out = append(out, ev)
		}
		fh.Close()
	}
	return out
}

func waitFor(t *testing.T, dir string, n int) []map[string]any {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		evs := readEvents(t, dir)
		if len(evs) >= n || time.Now().After(deadline) {
			return evs
		}
		time.Sleep(10 * time.Millisecond)
	}
}

type fakeCtx struct{ m map[string]any }

func (f fakeCtx) Get(k string) (any, bool) { v, ok := f.m[k]; return v, ok }

func TestNilFeedIsNoop(t *testing.T) {
	var f *Feed
	tr := f.Open("h1", map[string]any{"x": 1})
	if tr != nil {
		t.Fatal("disabled feed must return a nil turn")
	}
	tr.Piece(Piece{Content: "a"})
	tr.Out([]byte("{}\n"))
	tr.Finish(nil)
	tr.Close()
	OutFrom(fakeCtx{m: map[string]any{}}, []byte("x"))
	if f.Dropped() != 0 || f.Enabled() {
		t.Fatal("nil feed must report disabled, zero drops")
	}
}

func TestTurnEventsInOrderAndVerbatim(t *testing.T) {
	dir := t.TempDir()
	f := New(dir)
	prompt := "<|im_start|>system\nyou are <think>\nkid\n<|im_end|>\n\"quoted\" & <b>"
	tr := f.Open("harbor123", map[string]any{"rendered_prompt": prompt, "model": "m"})
	tr.Piece(Piece{Content: "Hel", DraftN: 3, DraftAcceptedN: 1})
	tr.Piece(Piece{Content: "lo\n"})
	cached := 7
	tr.Piece(Piece{Content: "", Done: true, DoneReason: "stop", EvalCount: 2, PromptEvalCount: 10, PromptEvalCachedCount: &cached})
	ctx := fakeCtx{m: map[string]any{ContextKey: tr}}
	OutFrom(ctx, []byte(`{"message":{"content":"Hel"}}`+"\n"))
	tr.Finish(nil, nil)
	tr.Close()

	evs := waitFor(t, dir, 6)
	if len(evs) != 6 {
		t.Fatalf("want 6 events, got %d", len(evs))
	}
	kinds := []string{}
	for _, e := range evs {
		kinds = append(kinds, e["ev"].(string))
		if e["schema"] != Schema {
			t.Fatalf("schema missing on %v", e)
		}
		if e["turn"] != "harbor123" {
			t.Fatalf("harbor turn id missing on %v", e)
		}
	}
	if strings.Join(kinds, ",") != "open,piece,piece,piece,out,close" {
		t.Fatalf("order: %v", kinds)
	}
	if evs[0]["rendered_prompt"] != prompt {
		t.Fatalf("rendered prompt not verbatim: %q", evs[0]["rendered_prompt"])
	}
	// least processed: no HTML escaping on disk
	raw, _ := os.ReadFile(filepath.Join(dir, time.Now().UTC().Format("2006-01-02"), "ollama-feed-"+time.Now().UTC().Format("2006-01-02")+".jsonl"))
	if strings.Contains(string(raw), `<`) {
		t.Fatal("feed must not HTML-escape the prompt")
	}
	if evs[1]["seq"].(float64) != 1 || evs[3]["seq"].(float64) != 3 || evs[1]["draft_n"].(float64) != 3 {
		t.Fatalf("piece seq/draft wrong: %v %v", evs[1], evs[3])
	}
	if evs[3]["done"] != true || evs[3]["done_reason"] != "stop" {
		t.Fatalf("done piece: %v", evs[3])
	}
	if evs[4]["raw"] != `{"message":{"content":"Hel"}}`+"\n" {
		t.Fatalf("out not verbatim: %v", evs[4])
	}
	c := evs[5]
	if c["pieces"].(float64) != 3 || c["outs"].(float64) != 1 || c["eval_count"].(float64) != 2 ||
		c["prompt_eval_count"].(float64) != 10 || c["prompt_eval_cached_count"].(float64) != 7 ||
		c["finished"] != true || c["done_reason"] != "stop" {
		t.Fatalf("close: %v", c)
	}
}

func TestBackpressureDropsAndCounts(t *testing.T) {
	// A feed whose writer never starts (directory unwritable) still never blocks the caller.
	f := New(filepath.Join(string([]byte{0}), "nope"))
	f.once.Do(func() {}) // swallow the writer start: the queue fills, sends must not block
	tr := f.Open("h", nil)
	done := make(chan struct{})
	go func() {
		for i := 0; i < queueSize+100; i++ {
			tr.Piece(Piece{Content: "x"})
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("feed blocked generation")
	}
	if f.Dropped() < 100 {
		t.Fatalf("drops not counted: %d", f.Dropped())
	}
}

func TestErrorsLandInClose(t *testing.T) {
	dir := t.TempDir()
	f := New(dir)
	tr := f.Open("", nil)
	tr.Finish(os.ErrClosed)
	tr.Close()
	evs := waitFor(t, dir, 2)
	c := evs[len(evs)-1]
	errs, _ := c["errors"].([]any)
	if len(errs) != 1 || !strings.Contains(errs[0].(string), "closed") {
		t.Fatalf("close errors: %v", c)
	}
	if _, ok := c["done_reason"]; !ok {
		t.Fatalf("close must carry done_reason (null when no final piece): %v", c)
	}
}
