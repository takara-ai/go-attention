// main.go
package main

import (
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/takara-ai/go-attention/attention"
)

// Document holds a title, some content, and an embedding vector.
type Document struct {
	Title     string
	Content   string
	Embedding attention.Vector
}

// parseVector converts a comma-separated string (e.g. "1.0,0.0,1.0,0.0")
// into an attention.Vector.
func parseVector(s string) (attention.Vector, error) {
	parts := strings.Split(s, ",")
	vec := make(attention.Vector, len(parts))
	for i, p := range parts {
		f, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return nil, err
		}
		vec[i] = f
	}
	return vec, nil
}

func main() {
	// Define a small "database" of documents with 4-dimensional embeddings.
	documents := []Document{
		{
			Title:     "Cats",
			Content:   "Cats are small, carnivorous mammals that are often valued by humans for companionship.",
			Embedding: attention.Vector{1.0, 0.0, 1.0, 0.0},
		},
		{
			Title:     "Dogs",
			Content:   "Dogs are domesticated mammals, known for their loyalty and companionship with humans.",
			Embedding: attention.Vector{0.0, 1.0, 0.0, 1.0},
		},
		{
			Title:     "Neutral",
			Content:   "This document does not lean toward any particular subject.",
			Embedding: attention.Vector{0.5, 0.5, 0.5, 0.5},
		},
		{
			Title:     "Birds",
			Content:   "Birds are warm-blooded vertebrates characterized by feathers and beaks.",
			Embedding: attention.Vector{1.0, 1.0, 0.0, 0.0},
		},
	}

	// Default query vector (for example, looking for cat‑like features).
	defaultQuery := attention.Vector{1.0, 0.0, 1.0, 0.0}

	// If the user provides a query vector as the first command-line argument,
	// parse it. Otherwise, use the default.
	var query attention.Vector
	var err error
	if len(os.Args) > 1 {
		query, err = parseVector(os.Args[1])
		if err != nil {
			log.Fatalf("Error parsing query vector: %v", err)
		}
	} else {
		query = defaultQuery
	}

	// Ensure that the query vector dimension matches our document embeddings.
	if len(query) != len(documents[0].Embedding) {
		log.Fatalf("Query vector dimension (%d) does not match document embedding dimension (%d)",
			len(query), len(documents[0].Embedding))
	}

	// Build the keys and values matrices from the document embeddings.
	// In this simple example, we use the embeddings for both keys and values.
	keys := make(attention.Matrix, len(documents))
	values := make(attention.Matrix, len(documents))
	for i, doc := range documents {
		keys[i] = doc.Embedding
		values[i] = doc.Embedding
	}

	// Compute dot-product attention.
	// The returned 'weights' slice contains the attention weight for each document.
	_, weights, err := attention.DotProductAttention(query, keys, values)
	if err != nil {
		log.Fatalf("Error computing attention: %v", err)
	}

	// Create a slice that holds each document's index and its corresponding attention score.
	type docScore struct {
		index int
		score float64
	}
	scores := make([]docScore, len(documents))
	for i, w := range weights {
		scores[i] = docScore{index: i, score: w}
	}

	// Sort documents by descending attention weight (i.e. relevance).
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Print the query and the documents ranked by their relevance.
	fmt.Println("Query Vector:", query)
	fmt.Println("\nDocument Relevance Scores:")
	for _, ds := range scores {
		doc := documents[ds.index]
		fmt.Printf("Title: %s\nScore: %.3f\nContent: %s\n\n", doc.Title, ds.score, doc.Content)
	}
}
