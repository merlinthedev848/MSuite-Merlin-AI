package main

import (
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/bcrypt"
	_ "modernc.org/sqlite"
)

var (
	db     *sql.DB
	dbOnce sync.Once
)

type User struct {
	ID          int64     `json:"id"`
	Username    string    `json:"username"`
	DisplayName string    `json:"display_name"`
	IsAdmin     bool      `json:"is_admin"`
	CreatedAt   time.Time `json:"created_at"`
}

type Session struct {
	ID        string    `json:"id"`
	Title     string    `json:"title"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	UserID    *int64    `json:"user_id,omitempty"`
}

type Message struct {
	ID        int64     `json:"id"`
	SessionID string    `json:"session_id"`
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	CreatedAt time.Time `json:"created_at"`
}

type Todo struct {
	ID        int64     `json:"id"`
	Task      string    `json:"task"`
	Completed bool      `json:"completed"`
	Priority  string    `json:"priority"`
	CreatedAt time.Time `json:"created_at"`
}

type Note struct {
	ID        int64     `json:"id"`
	Title     string    `json:"title"`
	Content   string    `json:"content"`
	Tags      string    `json:"tags"`
	CreatedAt time.Time `json:"created_at"`
}

type KnowledgeGap struct {
	ID        int64  `json:"id"`
	Question  string `json:"question"`
	Query     string `json:"query"` // Alias for JSON compatibility
	Context   string `json:"context"`
	Status    string `json:"status"`
	Resolved  bool   `json:"resolved"`
	CreatedAt string `json:"created_at"`
}

type KnowledgeEntry struct {
	ID          int64   `json:"id"`
	Topic       string  `json:"topic"`
	Lesson      string  `json:"lesson"`
	CodeExample string  `json:"code_example"`
	Score       float64 `json:"score,omitempty"`
}

type CalendarEvent struct {
	ID          int64     `json:"id"`
	Title       string    `json:"title"`
	StartTime   string    `json:"start_time"`
	EndTime     string    `json:"end_time"`
	Description string    `json:"description"`
	CreatedAt   time.Time `json:"created_at"`
	UserID      *int64    `json:"user_id,omitempty"`
}

type AuditLogEntry struct {
	ID        int64     `json:"id"`
	Action    string    `json:"action"`
	Details   string    `json:"details"`
	Level     string    `json:"level"`
	CreatedAt time.Time `json:"created_at"`
}

func initDatabase(dbPath string) (*sql.DB, error) {
	var err error
	dbOnce.Do(func() {
		dsn := fmt.Sprintf("%s?_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)&_pragma=busy_timeout(30000)&_pragma=foreign_keys(ON)", dbPath)
		db, err = sql.Open("sqlite", dsn)
		if err != nil {
			return
		}
		db.SetMaxOpenConns(25)
		db.SetMaxIdleConns(10)
		db.SetConnMaxLifetime(time.Hour)

		schema := `
		CREATE TABLE IF NOT EXISTS users (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			username TEXT UNIQUE NOT NULL,
			password_hash TEXT NOT NULL,
			display_name TEXT NOT NULL,
			is_admin INTEGER NOT NULL DEFAULT 0,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);

		CREATE TABLE IF NOT EXISTS user_tokens (
			token TEXT PRIMARY KEY,
			user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			expires_at DATETIME NOT NULL
		);

		CREATE TABLE IF NOT EXISTS sessions (
			id TEXT PRIMARY KEY,
			title TEXT NOT NULL DEFAULT 'New Conversation',
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			user_id INTEGER REFERENCES users(id) ON DELETE CASCADE
		);

		CREATE TABLE IF NOT EXISTS messages (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
			role TEXT NOT NULL,
			content TEXT NOT NULL,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);

		CREATE TABLE IF NOT EXISTS todos (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			task TEXT NOT NULL,
			completed INTEGER NOT NULL DEFAULT 0,
			priority TEXT NOT NULL DEFAULT 'normal',
			user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);

		CREATE TABLE IF NOT EXISTS notes (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			title TEXT NOT NULL,
			content TEXT NOT NULL,
			tags TEXT NOT NULL DEFAULT '',
			user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);

		CREATE TABLE IF NOT EXISTS gaps (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			query TEXT NOT NULL,
			context TEXT NOT NULL DEFAULT '',
			resolved INTEGER NOT NULL DEFAULT 0,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);

		CREATE TABLE IF NOT EXISTS knowledge (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			topic TEXT NOT NULL,
			lesson TEXT NOT NULL,
			code_example TEXT NOT NULL DEFAULT '',
			embedding_json TEXT DEFAULT '',
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);

		CREATE TABLE IF NOT EXISTS audit_logs (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			action TEXT NOT NULL,
			details TEXT NOT NULL,
			level TEXT NOT NULL DEFAULT 'info',
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);

		CREATE TABLE IF NOT EXISTS calendar_events (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			title TEXT NOT NULL,
			start_time TEXT NOT NULL,
			end_time TEXT NOT NULL,
			description TEXT NOT NULL DEFAULT '',
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			user_id INTEGER REFERENCES users(id) ON DELETE CASCADE
		);

		CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
		CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
		CREATE INDEX IF NOT EXISTS idx_tokens_expires ON user_tokens(expires_at);
		`
		_, err = db.Exec(schema)
	})
	return db, err
}

// User Authentication
func RegisterUser(username, password, displayName string) (*User, string, error) {
	username = strings.TrimSpace(username)
	if username == "" || len(password) < 4 {
		return nil, "", fmt.Errorf("invalid username or password (minimum 4 characters)")
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return nil, "", err
	}

	if displayName == "" {
		displayName = username
	}

	var count int
	_ = db.QueryRow("SELECT COUNT(*) FROM users").Scan(&count)
	isAdmin := 0
	if count == 0 {
		isAdmin = 1 // First user is Admin
	}

	res, err := db.Exec(
		"INSERT INTO users (username, password_hash, display_name, is_admin) VALUES (?, ?, ?, ?)",
		username, string(hash), displayName, isAdmin,
	)
	if err != nil {
		return nil, "", fmt.Errorf("username already exists")
	}

	userID, _ := res.LastInsertId()
	token, err := createSessionToken(userID)
	if err != nil {
		return nil, "", err
	}

	u := &User{
		ID:          userID,
		Username:    username,
		DisplayName: displayName,
		IsAdmin:     isAdmin == 1,
		CreatedAt:   time.Now(),
	}
	return u, token, nil
}

func AuthenticateUser(username, password string) (*User, string, error) {
	var u User
	var hash string
	var isAdminInt int
	var createdAtStr string

	err := db.QueryRow(
		"SELECT id, username, password_hash, display_name, is_admin, created_at FROM users WHERE username = ?",
		strings.TrimSpace(username),
	).Scan(&u.ID, &u.Username, &hash, &u.DisplayName, &isAdminInt, &createdAtStr)
	if err != nil {
		return nil, "", fmt.Errorf("invalid username or password")
	}

	if err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(password)); err != nil {
		return nil, "", fmt.Errorf("invalid username or password")
	}

	u.IsAdmin = (isAdminInt == 1)
	token, err := createSessionToken(u.ID)
	if err != nil {
		return nil, "", err
	}
	return &u, token, nil
}

func GetUserByToken(token string) (*User, error) {
	if token == "" {
		return nil, fmt.Errorf("missing token")
	}

	var u User
	var isAdminInt int
	var createdAtStr string
	err := db.QueryRow(`
		SELECT u.id, u.username, u.display_name, u.is_admin, u.created_at
		FROM users u
		JOIN user_tokens t ON u.id = t.user_id
		WHERE t.token = ? AND t.expires_at > CURRENT_TIMESTAMP
	`, token).Scan(&u.ID, &u.Username, &u.DisplayName, &isAdminInt, &createdAtStr)
	if err != nil {
		return nil, err
	}
	u.IsAdmin = (isAdminInt == 1)
	return &u, nil
}

func DeleteUserToken(token string) error {
	_, err := db.Exec("DELETE FROM user_tokens WHERE token = ?", token)
	return err
}

func createSessionToken(userID int64) (string, error) {
	bytes := make([]byte, 32)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	token := hex.EncodeToString(bytes)
	expires := time.Now().Add(30 * 24 * time.Hour)
	_, err := db.Exec(
		"INSERT INTO user_tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
		token, userID, expires,
	)
	return token, err
}

// Sessions & Messages
func CreateSession(id, title string, userID *int64) (*Session, error) {
	if id == "" {
		bytes := make([]byte, 16)
		_, _ = rand.Read(bytes)
		id = hex.EncodeToString(bytes)
	}
	if title == "" {
		title = "New Conversation"
	}
	now := time.Now()
	_, err := db.Exec(
		"INSERT INTO sessions (id, title, user_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
		id, title, userID, now, now,
	)
	if err != nil {
		return nil, err
	}
	return &Session{ID: id, Title: title, UserID: userID, CreatedAt: now, UpdatedAt: now}, nil
}

func GetSessions(userID *int64, limit int) ([]Session, error) {
	if limit <= 0 {
		limit = 50
	}
	var rows *sql.Rows
	var err error
	if userID != nil {
		rows, err = db.Query(
			"SELECT id, title, created_at, updated_at, user_id FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
			*userID, limit,
		)
	} else {
		rows, err = db.Query(
			"SELECT id, title, created_at, updated_at, user_id FROM sessions WHERE user_id IS NULL ORDER BY updated_at DESC LIMIT ?",
			limit,
		)
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var sessions []Session
	for rows.Next() {
		var s Session
		var createdStr, updatedStr string
		if err := rows.Scan(&s.ID, &s.Title, &createdStr, &updatedStr, &s.UserID); err != nil {
			continue
		}
		sessions = append(sessions, s)
	}
	return sessions, nil
}

func GetSessionMessages(sessionID string) ([]Message, error) {
	rows, err := db.Query(
		"SELECT id, session_id, role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
		sessionID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var msgs []Message
	for rows.Next() {
		var m Message
		var createdStr string
		if err := rows.Scan(&m.ID, &m.SessionID, &m.Role, &m.Content, &createdStr); err != nil {
			continue
		}
		msgs = append(msgs, m)
	}
	return msgs, nil
}

func SaveMessage(sessionID, role, content string) error {
	_, err := db.Exec(
		"INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
		sessionID, role, content,
	)
	if err == nil {
		_, _ = db.Exec("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", sessionID)
	}
	return err
}

func DeleteSession(sessionID string) error {
	_, err := db.Exec("DELETE FROM sessions WHERE id = ?", sessionID)
	return err
}

// Personal Assistant: Todos & Notes
func GetTodos(userID *int64) ([]Todo, error) {
	rows, err := db.Query("SELECT id, task, completed, priority, created_at FROM todos ORDER BY id DESC")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var todos []Todo
	for rows.Next() {
		var t Todo
		var compInt int
		var createdStr string
		if err := rows.Scan(&t.ID, &t.Task, &compInt, &t.Priority, &createdStr); err != nil {
			continue
		}
		t.Completed = (compInt == 1)
		todos = append(todos, t)
	}
	return todos, nil
}

func AddTodo(task, priority string, userID *int64) (*Todo, error) {
	if priority == "" {
		priority = "normal"
	}
	res, err := db.Exec("INSERT INTO todos (task, priority) VALUES (?, ?)", task, priority)
	if err != nil {
		return nil, err
	}
	id, _ := res.LastInsertId()
	return &Todo{ID: id, Task: task, Priority: priority, Completed: false, CreatedAt: time.Now()}, nil
}

func ToggleTodo(id int64) (bool, error) {
	var comp int
	err := db.QueryRow("SELECT completed FROM todos WHERE id = ?", id).Scan(&comp)
	if err != nil {
		return false, err
	}
	newVal := 1 - comp
	_, err = db.Exec("UPDATE todos SET completed = ? WHERE id = ?", newVal, id)
	return newVal == 1, err
}

func DeleteTodo(id int64) error {
	_, err := db.Exec("DELETE FROM todos WHERE id = ?", id)
	return err
}

func GetNotes(userID *int64, search string) ([]Note, error) {
	query := "SELECT id, title, content, tags, updated_at FROM notes "
	var args []any
	if search != "" {
		query += "WHERE (title LIKE ? OR content LIKE ? OR tags LIKE ?) "
		pattern := "%" + search + "%"
		args = append(args, pattern, pattern, pattern)
	}
	query += "ORDER BY id DESC"

	rows, err := db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var notes []Note
	for rows.Next() {
		var n Note
		var updatedStr string
		if err := rows.Scan(&n.ID, &n.Title, &n.Content, &n.Tags, &updatedStr); err != nil {
			continue
		}
		notes = append(notes, n)
	}
	return notes, nil
}

func AddNote(title, content, tags string, userID *int64) (*Note, error) {
	res, err := db.Exec("INSERT INTO notes (title, content, tags, user_id) VALUES (?, ?, ?, ?)", title, content, tags, userID)
	if err != nil {
		return nil, err
	}
	id, _ := res.LastInsertId()
	return &Note{ID: id, Title: title, Content: content, Tags: tags, CreatedAt: time.Now()}, nil
}

func DeleteNote(id int64) error {
	_, err := db.Exec("DELETE FROM notes WHERE id = ?", id)
	return err
}

// Knowledge Gaps
func GetKnowledgeGaps(unresolvedOnly bool) ([]KnowledgeGap, error) {
	q := "SELECT id, question, context, status, created_at FROM gaps "
	if unresolvedOnly {
		q += "WHERE status != 'resolved' "
	}
	q += "ORDER BY id DESC"

	rows, err := db.Query(q)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var gaps []KnowledgeGap
	for rows.Next() {
		var g KnowledgeGap
		var createdStr string
		if err := rows.Scan(&g.ID, &g.Question, &g.Context, &g.Status, &createdStr); err != nil {
			continue
		}
		g.Query = g.Question
		g.Resolved = (g.Status == "resolved")
		g.CreatedAt = createdStr
		gaps = append(gaps, g)
	}
	return gaps, nil
}

func AddKnowledgeGap(query, context string) error {
	_, err := db.Exec("INSERT INTO gaps (question, context, status) VALUES (?, ?, 'pending')", query, context)
	return err
}

func DeleteKnowledgeGap(id int64) error {
	_, err := db.Exec("DELETE FROM gaps WHERE id = ?", id)
	return err
}

func ClaimSession(sessionID string, userID int64) error {
	_, err := db.Exec("UPDATE sessions SET user_id = ? WHERE id = ?", userID, sessionID)
	return err
}


// Live Hardware Telemetry
func GetSystemTelemetry() map[string]any {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	uptimeSeconds := int(time.Since(startTime).Seconds())
	hours := uptimeSeconds / 3600
	minutes := (uptimeSeconds % 3600) / 60

	var knowledgeCount, todoCount, noteCount, gapCount int
	_ = db.QueryRow("SELECT COUNT(*) FROM knowledge").Scan(&knowledgeCount)
	_ = db.QueryRow("SELECT COUNT(*) FROM todos WHERE completed = 0").Scan(&todoCount)
	_ = db.QueryRow("SELECT COUNT(*) FROM notes").Scan(&noteCount)
	_ = db.QueryRow("SELECT COUNT(*) FROM gaps WHERE resolved = 0").Scan(&gapCount)

	hostname, _ := os.Hostname()

	return map[string]any{
		"platform":         fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
		"hostname":         hostname,
		"logical_cores":    runtime.NumCPU(),
		"goroutines":       runtime.NumGoroutine(),
		"allocated_memory": fmt.Sprintf("%.2f MB", float64(m.Alloc)/(1024*1024)),
		"total_sys_memory": fmt.Sprintf("%.2f MB", float64(m.Sys)/(1024*1024)),
		"gc_cycles":        m.NumGC,
		"uptime_seconds":   uptimeSeconds,
		"uptime_formatted": fmt.Sprintf("%dh %dm", hours, minutes),
		"database_metrics": map[string]int{
			"knowledge_entries": knowledgeCount,
			"pending_todos":     todoCount,
			"notes":             noteCount,
			"unresolved_gaps":   gapCount,
		},
		"engine_mode": "Compiled Native Linux Go Server",
		"status":      "operational",
	}
}

// Log audit events
func LogAuditEvent(action, details, level string) {
	if level == "" {
		level = "info"
	}
	_, _ = db.Exec("INSERT INTO audit_logs (action, details, level) VALUES (?, ?, ?)", action, details, level)
}

func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}
	var dot, normA, normB float64
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0.0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}


// Calendar Management
func GetCalendarEvents(limit int) ([]CalendarEvent, error) {
	if limit <= 0 {
		limit = 50
	}
	rows, err := db.Query("SELECT id, title, start_time, end_time, description, created_at FROM calendar_events ORDER BY start_time ASC LIMIT ?", limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var events []CalendarEvent
	for rows.Next() {
		var e CalendarEvent
		if err := rows.Scan(&e.ID, &e.Title, &e.StartTime, &e.EndTime, &e.Description, &e.CreatedAt); err == nil {
			events = append(events, e)
		}
	}
	if events == nil {
		events = []CalendarEvent{}
	}
	return events, nil
}

func BookCalendarEvent(title, startTime, endTime, desc string, userID *int64) (*CalendarEvent, error) {
	title = strings.TrimSpace(title)
	if title == "" {
		title = "Appointment"
	}
	now := time.Now()
	res, err := db.Exec(
		"INSERT INTO calendar_events (title, start_time, end_time, description, created_at) VALUES (?, ?, ?, ?, ?)",
		title, startTime, endTime, desc, now,
	)
	if err != nil {
		return nil, err
	}
	id, _ := res.LastInsertId()
	LogAuditEvent("CALENDAR_BOOK", fmt.Sprintf("Booked '%s' (%s to %s)", title, startTime, endTime), "info")
	return &CalendarEvent{
		ID:          id,
		Title:       title,
		StartTime:   startTime,
		EndTime:     endTime,
		Description: desc,
		CreatedAt:   now,
		UserID:      userID,
	}, nil
}

func CheckCalendarAvailability(startTime, endTime string) (bool, string, error) {
	var count int
	err := db.QueryRow(
		"SELECT COUNT(*) FROM calendar_events WHERE (start_time < ? AND end_time > ?)",
		endTime, startTime,
	).Scan(&count)
	if err != nil {
		return false, "", err
	}
	if count > 0 {
		return false, "Slot conflicts with an existing scheduled event.", nil
	}
	return true, "Time slot is completely clear and available.", nil
}

func DeleteCalendarEvent(id int64) error {
	_, err := db.Exec("DELETE FROM calendar_events WHERE id = ?", id)
	if err == nil {
		LogAuditEvent("CALENDAR_DELETE", fmt.Sprintf("Deleted calendar event #%d", id), "info")
	}
	return err
}

// Knowledge Base Query & Mutate
func GetKnowledge(query string, limit int) ([]KnowledgeEntry, error) {
	if limit <= 0 {
		limit = 50
	}
	var rows *sql.Rows
	var err error
	if query == "" {
		rows, err = db.Query("SELECT id, topic, lesson, code_example FROM knowledge ORDER BY id DESC LIMIT ?", limit)
	} else {
		pattern := "%" + query + "%"
		rows, err = db.Query("SELECT id, topic, lesson, code_example FROM knowledge WHERE topic LIKE ? OR lesson LIKE ? ORDER BY id DESC LIMIT ?", pattern, pattern, limit)
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var entries []KnowledgeEntry
	for rows.Next() {
		var k KnowledgeEntry
		if err := rows.Scan(&k.ID, &k.Topic, &k.Lesson, &k.CodeExample); err == nil {
			entries = append(entries, k)
		}
	}
	if entries == nil {
		entries = []KnowledgeEntry{}
	}
	return entries, nil
}

func AddKnowledge(topic, lesson, codeExample string) (*KnowledgeEntry, error) {
	topic = strings.TrimSpace(topic)
	lesson = strings.TrimSpace(lesson)
	if topic == "" || lesson == "" {
		return nil, fmt.Errorf("topic and lesson are required")
	}
	res, err := db.Exec(
		"INSERT INTO knowledge (topic, lesson, code_example) VALUES (?, ?, ?)",
		topic, lesson, codeExample,
	)
	if err != nil {
		return nil, err
	}
	id, _ := res.LastInsertId()
	LogAuditEvent("KNOWLEDGE_ADD", fmt.Sprintf("Added knowledge entry on '%s'", topic), "info")
	return &KnowledgeEntry{
		ID:          id,
		Topic:       topic,
		Lesson:      lesson,
		CodeExample: codeExample,
	}, nil
}

func DeleteKnowledge(id int64) error {
	_, err := db.Exec("DELETE FROM knowledge WHERE id = ?", id)
	if err == nil {
		LogAuditEvent("KNOWLEDGE_DELETE", fmt.Sprintf("Deleted knowledge entry #%d", id), "info")
	}
	return err
}

// Audit Logs
func GetAuditLogs(limit int) ([]AuditLogEntry, error) {
	if limit <= 0 {
		limit = 50
	}
	rows, err := db.Query("SELECT id, action, details, level, created_at FROM audit_logs ORDER BY id DESC LIMIT ?", limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var logs []AuditLogEntry
	for rows.Next() {
		var l AuditLogEntry
		if err := rows.Scan(&l.ID, &l.Action, &l.Details, &l.Level, &l.CreatedAt); err == nil {
			logs = append(logs, l)
		}
	}
	if logs == nil {
		logs = []AuditLogEntry{}
	}
	return logs, nil
}

// Database Vacuum & Maintenance
func VacuumDatabase() (int64, error) {
	_, err := db.Exec("PRAGMA wal_checkpoint(TRUNCATE)")
	if err != nil {
		return 0, err
	}
	_, err = db.Exec("VACUUM")
	if err != nil {
		return 0, err
	}
	LogAuditEvent("DATABASE_VACUUM", "Executed WAL checkpoint and VACUUM operation", "info")
	return 64, nil
}
