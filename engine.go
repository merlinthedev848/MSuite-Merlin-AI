package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

var (
	ollamaURL = "http://localhost:11434"
	startTime = time.Now()
)

func init() {
	if u := os.Getenv("OLLAMA_URL"); u != "" {
		ollamaURL = strings.TrimRight(u, "/")
	}
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OllamaChatRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
	Options  map[string]any `json:"options,omitempty"`
}

type OllamaChatResponse struct {
	Model     string      `json:"model"`
	CreatedAt string      `json:"created_at"`
	Message   ChatMessage `json:"message"`
	Done      bool        `json:"done"`
}

type ModelInfo struct {
	Name       string `json:"name"`
	Model      string `json:"model"`
	ModifiedAt string `json:"modified_at"`
	Size       int64  `json:"size"`
}

type OllamaTagsResponse struct {
	Models []ModelInfo `json:"models"`
}

// GetAvailableModels queries local Ollama instance for installed models
func GetAvailableModels() []string {
	client := http.Client{Timeout: 800 * time.Millisecond}
	resp, err := client.Get(ollamaURL + "/api/tags")
	if err != nil {
		return []string{}
	}
	defer resp.Body.Close()

	var tags OllamaTagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		return []string{}
	}

	var names []string
	for _, m := range tags.Models {
		names = append(names, m.Name)
	}
	return names
}

// StreamResponse streams tokens from local Ollama or returns deterministic fast paths immediately
func StreamResponse(ctx context.Context, prompt, sessionID, activeModel string, flusher http.Flusher, w io.Writer) (string, error) {
	promptClean := strings.TrimSpace(prompt)
	promptLower := strings.ToLower(promptClean)
	cleaned := strings.Trim(promptLower, "?.! ,")

	// =========================================================================
	// DETERMINISTIC FAST-PATH 1: Real-Time Local Clock & Date (0.05ms)
	// =========================================================================
	timeTriggers := []string{
		"what time is it", "what's the time", "whats the time", "tell me the time",
		"current time", "what is the time", "the time please", "time check",
		"what day is today", "what day is it", "what is the date", "what date is it",
		"today's date", "todays date", "time please", "do you know the time",
		"tell me the date", "date and time", "time and date", "what time",
	}
	isTime := false
	if cleaned == "time" || cleaned == "date" || cleaned == "clock" || cleaned == "what time" {
		isTime = true
	} else {
		for _, t := range timeTriggers {
			if strings.Contains(promptLower, t) {
				isTime = true
				break
			}
		}
	}

	if isTime {
		now := time.Now()
		formattedTime := now.Format("3:04 PM")
		formattedDate := now.Format("Monday, January 2, 2006")
		resp := fmt.Sprintf("The current time is **%s** on %s, Sir.", formattedTime, formattedDate)
		sendSSE(w, flusher, resp)
		return resp, nil
	}

	// =========================================================================
	// DETERMINISTIC FAST-PATH 2: Self-Diagnostics & Host Hardware Telemetry (0.1ms)
	// =========================================================================
	diagTriggers := []string{
		"self diagnostics", "self diagnostic", "run diagnostics", "run self diagnostics",
		"system diagnostics", "diagnostics", "diagnostic check", "diagnostic test",
		"system status", "telemetry report", "health check", "suit status",
		"do self diagnostics", "do a diagnostic check", "run a system check",
		"system check", "check diagnostics", "hardware status",
	}
	isDiag := false
	if cleaned == "diagnostics" || cleaned == "diagnostic" || cleaned == "status" || cleaned == "system status" {
		isDiag = true
	} else {
		for _, d := range diagTriggers {
			if strings.Contains(promptLower, d) {
				isDiag = true
				break
			}
		}
	}

	if isDiag {
		tele := GetSystemTelemetry()
		dbMetrics := tele["database_metrics"].(map[string]int)
		resp := fmt.Sprintf(
			"**Merlin Diagnostic Status Report** — *All Primary Systems Online & Calibrated*\n\n"+
				"- **Host Architecture**: `%s` (Node: `%s`)\n"+
				"- **Engine Runtime**: `%s` (Uptime: %s)\n"+
				"- **Concurrency**: `%d CPU Cores` / `%d Active Goroutines`\n"+
				"- **Memory Allocation**: `%s` Allocated / `%s` System Total\n"+
				"- **Knowledge Engine**: `%d Indexed Entries` (FTS5 + Dense Vector Engine)\n"+
				"- **Active Task Queue**: `%d Pending Tasks` | `%d Notes`\n"+
				"- **Synthesizer Engine**: `Google UK English Male` (Deep Baritone Persona)\n\n"+
				"All quantum circuits, Linux epoll pipelines, and neural processors operating at peak efficiency, Sir.",
			tele["platform"], tele["hostname"],
			tele["engine_mode"], tele["uptime_formatted"],
			tele["logical_cores"], tele["goroutines"],
			tele["allocated_memory"], tele["total_sys_memory"],
			dbMetrics["knowledge_entries"],
			dbMetrics["pending_todos"], dbMetrics["notes"],
		)
		sendSSE(w, flusher, resp)
		return resp, nil
	}

	// =========================================================================
	// DETERMINISTIC FAST-PATH 3: Conversational Greetings (<0.05ms)
	// =========================================================================
	greetings := []string{"hello", "hey", "hi", "good morning", "good evening", "who are you"}
	for _, g := range greetings {
		if cleaned == g || strings.HasPrefix(cleaned, g+" ") {
			resp := "Good day, Sir. Merlin neural intelligence online and standing by. How may I assist your directives?"
			sendSSE(w, flusher, resp)
			return resp, nil
		}
	}

	// =========================================================================
	// LIVE LLM INFERENCE (Ollama Native HTTP Streaming)
	// =========================================================================
	models := GetAvailableModels()
	if activeModel == "" || activeModel == "auto" {
		if len(models) > 0 {
			activeModel = models[0]
		} else {
			activeModel = "qwen2.5-coder:1.5b"
		}
	}

	// Build context from previous session messages
	prevMsgs, _ := GetSessionMessages(sessionID)
	var history []ChatMessage
	systemPrompt := "You are Merlin, an elite autonomous neural AI agent. Provide authoritative, concise, and production-grade responses. Your voice persona is Deep Baritone."
	history = append(history, ChatMessage{Role: "system", Content: systemPrompt})

	for _, m := range prevMsgs {
		history = append(history, ChatMessage{Role: m.Role, Content: m.Content})
	}
	history = append(history, ChatMessage{Role: "user", Content: promptClean})

	reqPayload := OllamaChatRequest{
		Model:    activeModel,
		Messages: history,
		Stream:   true,
		Options: map[string]any{
			"temperature": 0.3,
		},
	}

	payloadBytes, err := json.Marshal(reqPayload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", ollamaURL+"/api/chat", bytes.NewReader(payloadBytes))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	client := http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		// Ollama is offline fallback
		offlineMsg := "**Offline Intelligence Standby**\n\n" +
			"The compiled Linux Go server is active and running at native speed.\n\n" +
			"**To activate neural LLM inference on this Linux host:**\n" +
			"1. Run `ollama serve` in the background.\n" +
			"2. Run `ollama run qwen2.5-coder:1.5b`.\n\n" +
			"Merlin automatically connects to Ollama on `localhost:11434` with zero restart required."
		sendSSE(w, flusher, offlineMsg)
		return offlineMsg, nil
	}
	defer resp.Body.Close()

	var fullResponse strings.Builder
	reader := bufio.NewReader(resp.Body)

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			break
		}

		var chunk OllamaChatResponse
		if err := json.Unmarshal(line, &chunk); err == nil {
			if chunk.Message.Content != "" {
				fullResponse.WriteString(chunk.Message.Content)
				sendSSE(w, flusher, chunk.Message.Content)
			}
			if chunk.Done {
				break
			}
		}
	}

	return fullResponse.String(), nil
}

func sendSSE(w io.Writer, flusher http.Flusher, chunk string) {
	data, _ := json.Marshal(map[string]string{"chunk": chunk})
	fmt.Fprintf(w, "data: %s\n\n", data)
	if flusher != nil {
		flusher.Flush()
	}
}
