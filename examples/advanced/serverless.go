package main

import (
    "encoding/json"
    "log"
    "net/http"
    "sort"
    "strings"
    "sync"
    "time"
    "github.com/takara-ai/go-attention/attention"
)

// SearchRequest represents the incoming search query
type SearchRequest struct {
    Query string   `json:"query"`
    Docs  []string `json:"documents"`
}

// SearchResult represents a single matched document
type SearchResult struct {
    Text      string  `json:"text"`
    Score     float64 `json:"score"`
    Rank      int     `json:"rank"`
}

// SearchResponse represents the search results
type SearchResponse struct {
    Results []SearchResult `json:"results"`
    Timing float64        `json:"timing_ms"`
}

// Singleton pattern for embedder to avoid reinitialization
var (
    embedder     *SemanticEmbedder
    embedderOnce sync.Once
)

// SemanticEmbedder handles document embedding with attention
type SemanticEmbedder struct {
    dimension int
}

// Simple semantic embedding simulation
func (e *SemanticEmbedder) embedWord(word string) attention.Vector {
    word = strings.ToLower(word)
    embedding := make(attention.Vector, e.dimension)
    
    // Animal-related features
    if strings.Contains(word, "cat") || strings.Contains(word, "kitten") {
        embedding[0] = 1.0  // feline
        embedding[1] = 0.8  // pet
        embedding[2] = 0.6  // animal
    }
    if strings.Contains(word, "dog") || strings.Contains(word, "canine") {
        embedding[0] = 0.8  // animal
        embedding[1] = 0.8  // pet
        embedding[3] = 1.0  // canine
    }
    
    // Activity-related features
    if strings.Contains(word, "play") || strings.Contains(word, "chase") {
        embedding[4] = 1.0  // activity
        embedding[5] = 0.7  // movement
    }
    if strings.Contains(word, "sat") || strings.Contains(word, "love") {
        embedding[6] = 0.6  // state
    }
    
    // Tech-related features
    if strings.Contains(word, "ai") || strings.Contains(word, "artificial") || 
       strings.Contains(word, "intelligence") {
        embedding[8] = 1.0   // AI
        embedding[9] = 0.8   // technology
        embedding[10] = 0.7  // computing
    }
    if strings.Contains(word, "machine") || strings.Contains(word, "learning") {
        embedding[8] = 0.8   // AI
        embedding[9] = 0.9   // technology
        embedding[11] = 1.0  // learning
    }
    if strings.Contains(word, "neural") || strings.Contains(word, "network") {
        embedding[8] = 0.7   // AI
        embedding[9] = 0.8   // technology
        embedding[12] = 1.0  // networks
    }
    
    return embedding
}

// Embed a sentence into a fixed-size vector using attention
func (e *SemanticEmbedder) embedSentence(sentence string) attention.Vector {
    words := strings.Fields(sentence)
    if len(words) == 0 {
        return make(attention.Vector, e.dimension)
    }

    // Create embeddings for each word
    wordEmbeddings := make(attention.Matrix, len(words))
    for i, word := range words {
        wordEmbeddings[i] = e.embedWord(word)
    }

    // Use attention to combine word embeddings
    output, _, err := attention.DotProductAttention(wordEmbeddings[0], wordEmbeddings, wordEmbeddings)
    if err != nil {
        log.Printf("Error in embedSentence: %v", err)
        return make(attention.Vector, e.dimension)
    }

    return output
}

func getEmbedder() *SemanticEmbedder {
    embedderOnce.Do(func() {
        embedder = &SemanticEmbedder{
            dimension: 16,
        }
    })
    return embedder
}

// HandleSearch is the serverless entry point
func HandleSearch(w http.ResponseWriter, r *http.Request) {
    startTime := time.Now()

    if r.Method != http.MethodPost {
        http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
        return
    }

    // Parse request
    var req SearchRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request format", http.StatusBadRequest)
        return
    }

    // Get singleton embedder (zero cold start penalty after first request)
    emb := getEmbedder()

    // Process query and documents in parallel
    queryEmbed := emb.embedSentence(req.Query)
    
    // Use goroutines for parallel document processing
    docCount := len(req.Docs)
    docEmbeddings := make(attention.Matrix, docCount)
    var wg sync.WaitGroup
    wg.Add(docCount)
    
    for i := range req.Docs {
        go func(idx int) {
            defer wg.Done()
            docEmbeddings[idx] = emb.embedSentence(req.Docs[idx])
        }(i)
    }
    wg.Wait()

    // Compute attention scores
    _, weights, err := attention.DotProductAttention(queryEmbed, docEmbeddings, docEmbeddings)
    if err != nil {
        http.Error(w, "Processing error", http.StatusInternalServerError)
        return
    }

    // Prepare results
    results := make([]SearchResult, len(weights))
    for i := range weights {
        results[i] = SearchResult{
            Text:  req.Docs[i],
            Score: weights[i] * 100, // Convert to percentage
            Rank:  i + 1,
        }
    }

    // Sort results (quick sort is more efficient than bubble sort)
    sort.Slice(results, func(i, j int) bool {
        return results[i].Score > results[j].Score
    })

    // Return top results
    response := SearchResponse{
        Results: results[:min(3, len(results))],
        Timing:  float64(time.Since(startTime).Microseconds()) / 1000.0, // Convert to milliseconds
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

// For local testing
func main() {
    log.Printf("Starting semantic search server on :8080...")
    http.HandleFunc("/search", HandleSearch)
    log.Fatal(http.ListenAndServe(":8080", nil))
} 