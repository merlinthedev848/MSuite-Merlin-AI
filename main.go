package main

import (
	"compress/gzip"
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"
)

//go:embed web/index.html web/styles.css web/script.js
var embeddedWeb embed.FS

type ChatStreamRequest struct {
	Prompt    string `json:"prompt"`
	SessionID string `json:"session_id"`
	Mode      string `json:"mode"`
	Model     string `json:"model"`
}

type AuthRequest struct {
	Username    string `json:"username"`
	Password    string `json:"password"`
	DisplayName string `json:"display_name"`
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "5000"
	}
	dbPath := os.Getenv("DATABASE_PATH")
	if dbPath == "" {
		dbPath = "assistant_memory.db"
	}

	log.Printf("[Merlin Go] Initializing high-performance compiled server on port %s...", port)

	if _, err := initDatabase(dbPath); err != nil {
		log.Fatalf("[Merlin Go] Fatal: Failed to initialize SQLite database: %v", err)
	}
	log.Printf("[Merlin Go] SQLite WAL database operational: %s", dbPath)

	mux := http.NewServeMux()

	// Static Web Assets (Embedded in binary)
	webFS, err := fs.Sub(embeddedWeb, "web")
	if err != nil {
		log.Fatalf("[Merlin Go] Failed to mount embedded web directory: %v", err)
	}
	fileServer := http.FileServer(http.FS(webFS))

	// Static routes
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" || r.URL.Path == "/index.html" {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Header().Set("Cache-Control", "no-cache")
		} else if strings.HasSuffix(r.URL.Path, ".css") {
			w.Header().Set("Content-Type", "text/css; charset=utf-8")
			w.Header().Set("Cache-Control", "public, max-age=3600")
		} else if strings.HasSuffix(r.URL.Path, ".js") {
			w.Header().Set("Content-Type", "application/javascript; charset=utf-8")
			w.Header().Set("Cache-Control", "public, max-age=3600")
		}
		fileServer.ServeHTTP(w, r)
	})

	// API Health & Telemetry
	mux.HandleFunc("/api/health", handleHealth)
	mux.HandleFunc("/api/telemetry", handleTelemetry)

	// Chat Stream SSE
	mux.HandleFunc("/api/chat/stream", handleChatStream)

	// Auth API
	mux.HandleFunc("/api/auth/register", handleRegister)
	mux.HandleFunc("/api/auth/login", handleLogin)
	mux.HandleFunc("/api/auth/me", handleAuthMe)
	mux.HandleFunc("/api/auth/logout", handleLogout)

	// Sessions API
	mux.HandleFunc("/api/sessions", handleSessions)
	mux.HandleFunc("/api/sessions/", handleSessionItem)

	// Personal Assistant: Todos & Notes
	mux.HandleFunc("/api/todos", handleTodos)
	mux.HandleFunc("/api/todos/", handleTodoItem)
	mux.HandleFunc("/api/notes", handleNotes)
	mux.HandleFunc("/api/notes/", handleNoteItem)
	mux.HandleFunc("/api/gaps", handleGaps)

	// Models API
	mux.HandleFunc("/api/models", handleModels)

	handler := corsAndGzipMiddleware(mux)

	server := &http.Server{
		Addr:         ":" + port,
		Handler:      handler,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 180 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Graceful shutdown handling
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
		<-sigChan
		log.Println("[Merlin Go] Shutting down gracefully...")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(ctx)
		os.Exit(0)
	}()

	log.Printf("[Merlin Go] Quantum Core online at http://0.0.0.0:%s", port)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("[Merlin Go] Server error: %v", err)
	}
}

// Middleware: CORS & Gzip
type gzipResponseWriter struct {
	io.Writer
	http.ResponseWriter
}

func (w gzipResponseWriter) Write(b []byte) (int, error) {
	return w.Writer.Write(b)
}

func corsAndGzipMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		// Skip gzip on SSE streams
		if r.URL.Path == "/api/chat/stream" || !strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
			next.ServeHTTP(w, r)
			return
		}

		w.Header().Set("Content-Encoding", "gzip")
		gz := gzip.NewWriter(w)
		defer gz.Close()

		gzw := gzipResponseWriter{Writer: gz, ResponseWriter: w}
		next.ServeHTTP(gzw, r)
	})
}

// Route Handlers
func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	models := GetAvailableModels()
	backend := "ollama"
	if len(models) == 0 {
		backend = "offline_standby"
	}

	json.NewEncoder(w).Encode(map[string]any{
		"status":   "healthy",
		"engine":   "Merlin Go Native",
		"backend":  backend,
		"models":   models,
		"version":  "2.0.0-linux-native",
		"time_utc": time.Now().UTC().Format(time.RFC3339),
	})
}

func handleTelemetry(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(GetSystemTelemetry())
}

func handleChatStream(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatStreamRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	if req.Prompt == "" {
		http.Error(w, "Prompt is required", http.StatusBadRequest)
		return
	}

	if req.SessionID == "" {
		req.SessionID = "default"
	}

	// Prepare SSE response
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}
	flusher.Flush()

	// Save user message to database
	_ = SaveMessage(req.SessionID, "user", req.Prompt)

	// Stream AI response
	fullResp, err := StreamResponse(r.Context(), req.Prompt, req.SessionID, req.Model, flusher, w)
	if err == nil && fullResp != "" {
		_ = SaveMessage(req.SessionID, "assistant", fullResp)
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func handleRegister(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req AuthRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}
	user, token, err := RegisterUser(req.Username, req.Password, req.DisplayName)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]any{"error": err.Error()})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"user": user, "token": token, "logged_in": true})
}

func handleLogin(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req AuthRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}
	user, token, err := AuthenticateUser(req.Username, req.Password)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]any{"error": "Invalid username or password"})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"user": user, "token": token, "logged_in": true})
}

func handleAuthMe(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	token := extractBearerToken(r)
	if token == "" {
		json.NewEncoder(w).Encode(map[string]any{"logged_in": false})
		return
	}
	user, err := GetUserByToken(token)
	if err != nil {
		json.NewEncoder(w).Encode(map[string]any{"logged_in": false})
		return
	}
	json.NewEncoder(w).Encode(map[string]any{"logged_in": true, "user": user})
}

func handleLogout(w http.ResponseWriter, r *http.Request) {
	token := extractBearerToken(r)
	if token != "" {
		_ = DeleteUserToken(token)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"success": true})
}

func handleSessions(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	token := extractBearerToken(r)
	var userID *int64
	if user, err := GetUserByToken(token); err == nil {
		userID = &user.ID
	}

	if r.Method == http.MethodGet {
		sessions, err := GetSessions(userID, 50)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(sessions)
	} else if r.Method == http.MethodPost {
		var req struct {
			ID    string `json:"id"`
			Title string `json:"title"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		s, err := CreateSession(req.ID, req.Title, userID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(s)
	}
}

func handleSessionItem(w http.ResponseWriter, r *http.Request) {
	sessionID := strings.TrimPrefix(r.URL.Path, "/api/sessions/")
	if sessionID == "" {
		http.Error(w, "Session ID required", http.StatusBadRequest)
		return
	}

	// Check for messages subroute: /api/sessions/:id/messages
	if strings.HasSuffix(sessionID, "/messages") {
		sessionID = strings.TrimSuffix(sessionID, "/messages")
		w.Header().Set("Content-Type", "application/json")
		msgs, err := GetSessionMessages(sessionID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(msgs)
		return
	}

	if r.Method == http.MethodDelete {
		_ = DeleteSession(sessionID)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"success": true})
	}
}

func handleTodos(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	token := extractBearerToken(r)
	var userID *int64
	if user, err := GetUserByToken(token); err == nil {
		userID = &user.ID
	}

	if r.Method == http.MethodGet {
		todos, err := GetTodos(userID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(todos)
	} else if r.Method == http.MethodPost {
		var req struct {
			Task     string `json:"task"`
			Priority string `json:"priority"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		t, err := AddTodo(req.Task, req.Priority, userID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(t)
	}
}

func handleTodoItem(w http.ResponseWriter, r *http.Request) {
	idStr := strings.TrimPrefix(r.URL.Path, "/api/todos/")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		http.Error(w, "Invalid ID", http.StatusBadRequest)
		return
	}

	if r.Method == http.MethodPatch {
		newStatus, err := ToggleTodo(id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"id": id, "completed": newStatus})
	} else if r.Method == http.MethodDelete {
		_ = DeleteTodo(id)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"success": true})
	}
}

func handleNotes(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	token := extractBearerToken(r)
	var userID *int64
	if user, err := GetUserByToken(token); err == nil {
		userID = &user.ID
	}

	if r.Method == http.MethodGet {
		search := r.URL.Query().Get("q")
		notes, err := GetNotes(userID, search)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(notes)
	} else if r.Method == http.MethodPost {
		var req struct {
			Title   string `json:"title"`
			Content string `json:"content"`
			Tags    string `json:"tags"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		n, err := AddNote(req.Title, req.Content, req.Tags, userID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(n)
	}
}

func handleNoteItem(w http.ResponseWriter, r *http.Request) {
	idStr := strings.TrimPrefix(r.URL.Path, "/api/notes/")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		http.Error(w, "Invalid ID", http.StatusBadRequest)
		return
	}

	if r.Method == http.MethodDelete {
		_ = DeleteNote(id)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"success": true})
	}
}

func handleGaps(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method == http.MethodGet {
		gaps, err := GetKnowledgeGaps(true)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(gaps)
	} else if r.Method == http.MethodPost {
		var req struct {
			Query   string `json:"query"`
			Context string `json:"context"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		_ = AddKnowledgeGap(req.Query, req.Context)
		json.NewEncoder(w).Encode(map[string]any{"success": true})
	}
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"models": GetAvailableModels(),
	})
}

func extractBearerToken(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if strings.HasPrefix(auth, "Bearer ") {
		return strings.TrimPrefix(auth, "Bearer ")
	}
	return ""
}
