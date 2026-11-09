package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestMuxRouteAudit(t *testing.T) {
	_, _ = initDatabase("assistant_memory.db")

	endpoints := []struct {
		method string
		path   string
	}{
		{"POST", "/api/analyze/multimodal"},
		{"GET", "/api/audit"},
		{"POST", "/api/auth/claim_session"},
		{"POST", "/api/auth/logout"},
		{"GET", "/api/auth/me"},
		{"POST", "/api/calendar/availability"},
		{"POST", "/api/calendar/book"},
		{"POST", "/api/calendar/cancel"},
		{"POST", "/api/calendar/delete"},
		{"GET", "/api/calendar/events"},
		{"POST", "/api/chat/stop"},
		{"GET", "/api/conversations"},
		{"POST", "/api/dataset/import"},
		{"GET", "/api/dataset/popular"},
		{"GET", "/api/dataset/status"},
		{"GET", "/api/gaps"},
		{"POST", "/api/gaps/delete"},
		{"GET", "/api/knowledge"},
		{"POST", "/api/knowledge/delete"},
		{"POST", "/api/maintenance/vacuum"},
		{"GET", "/api/models"},
		{"POST", "/api/models/active"},
		{"GET", "/api/notes"},
		{"POST", "/api/notes/delete"},
		{"POST", "/api/notes/update"},
		{"GET", "/api/status"},
		{"POST", "/api/sync"},
		{"GET", "/api/system"},
		{"POST", "/api/teacher/argon/assimilate"},
		{"POST", "/api/teacher/argon/query"},
		{"GET", "/api/teacher/models"},
		{"GET", "/api/telephony/status"},
		{"POST", "/api/telephony/webhook"},
		{"GET", "/api/todos"},
		{"POST", "/api/todos/delete"},
		{"POST", "/api/todos/toggle"},
		{"POST", "/api/upload"},
	}

	mux := http.NewServeMux()
	registerRoutes(mux)

	for _, ep := range endpoints {
		req := httptest.NewRequest(ep.method, ep.path, strings.NewReader(`{"id": 1, "task": "test", "topic": "test", "query": "test"}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)

		// If a route fell back to the root static file handler "/", it returns 200 with HTML (or 404)
		contentType := w.Header().Get("Content-Type")
		if strings.Contains(contentType, "text/html") || w.Code == http.StatusNotFound {
			t.Errorf("FAILED ROUTE: %s %s returned status %d, Content-Type: %s (fell back to static handler!)",
				ep.method, ep.path, w.Code, contentType)
		} else {
			t.Logf("OK: %s %s -> status %d, Content-Type: %s", ep.method, ep.path, w.Code, contentType)
		}
	}
}
