package retrieval

import (
	"regexp"
	"sort"
	"strings"
)

type Document struct {
	Header   string `json:"header"`
	Link     string `json:"link"`
	Keywords string `json:"keywords"`
}

type DocumentScore struct {
	Document Document
	Score    float64
}

type Retriever struct {
	docs      []Document
	stopWords map[string]bool
}

func NewRetriever(docs []Document) *Retriever {
	// Русские и английские стоп-слова
	stopWords := map[string]bool{
		// Русские стоп-слова
		"и": true, "в": true, "во": true, "не": true, "что": true, "он": true, "на": true,
		"я": true, "с": true, "со": true, "как": true, "а": true, "то": true, "все": true,
		"она": true, "так": true, "его": true, "но": true, "да": true, "ты": true, "к": true,
		"у": true, "же": true, "вы": true, "за": true, "бы": true, "по": true, "только": true,
		"ее": true, "мне": true, "было": true, "вот": true, "от": true, "меня": true, "еще": true,
		"нет": true, "о": true, "из": true, "ему": true, "теперь": true, "когда": true, "даже": true,
		"ну": true, "вдруг": true, "ли": true, "если": true, "уже": true, "или": true, "ни": true,
		"быть": true, "был": true, "него": true, "до": true, "вас": true, "нибудь": true, "опять": true,
		"уж": true, "вам": true, "ведь": true, "там": true, "потом": true, "себя": true, "ничего": true,
		"ей": true, "может": true, "они": true, "тут": true, "где": true, "есть": true, "надо": true,
		"ней": true, "для": true, "мы": true, "тебя": true, "их": true, "чем": true, "была": true,
		"сам": true, "чтоб": true, "без": true, "будто": true, "чего": true, "раз": true, "тоже": true,
		"себе": true, "под": true, "будет": true, "ж": true, "тогда": true, "кто": true, "этот": true,
		"того": true, "потому": true, "этого": true, "какой": true, "совсем": true, "ним": true, "здесь": true,
		"этом": true, "один": true, "почти": true, "мой": true, "тем": true, "чтобы": true, "нее": true,
		"сейчас": true, "были": true, "куда": true, "зачем": true, "всех": true, "никогда": true, "можно": true,
		"при": true, "наконец": true, "два": true, "об": true, "другой": true, "хоть": true, "после": true,
		"над": true, "больше": true, "тот": true, "через": true, "эти": true, "нас": true, "про": true,
		"всего": true, "них": true, "какая": true, "много": true, "разве": true, "три": true, "эту": true,
		"моя": true, "впрочем": true, "хорошо": true, "свою": true, "этой": true, "перед": true, "иногда": true,
		"лучше": true, "чуть": true, "том": true, "нельзя": true, "такой": true, "им": true, "более": true,
		"всегда": true, "конечно": true, "всю": true, "между": true,

		// Английские стоп-слова
		"a": true, "an": true, "and": true, "are": true, "as": true, "at": true, "be": true, "by": true,
		"for": true, "from": true, "has": true, "he": true, "in": true, "is": true, "it": true,
		"its": true, "of": true, "on": true, "that": true, "the": true, "to": true, "was": true,
		"will": true, "with": true, "this": true, "but": true, "they": true, "have": true,
		"had": true, "what": true, "said": true, "each": true, "which": true, "she": true, "do": true,
		"how": true, "their": true, "if": true, "up": true, "out": true, "many": true, "then": true,
		"them": true, "these": true, "so": true, "some": true, "her": true, "would": true, "make": true,
		"like": true, "into": true, "him": true, "time": true, "two": true, "more": true, "go": true,
		"no": true, "way": true, "could": true, "my": true, "than": true, "first": true, "been": true,
		"call": true, "who": true, "oil": true, "sit": true, "now": true, "find": true, "down": true,
		"day": true, "did": true, "get": true, "come": true, "made": true, "may": true, "part": true,
	}

	return &Retriever{
		docs:      docs,
		stopWords: stopWords,
	}
}

func (r *Retriever) cleanQuery(query string) []string {
	// Удаляем знаки препинания и приводим к нижнему регистру
	reg := regexp.MustCompile(`[^\p{L}\s]+`)
	cleaned := reg.ReplaceAllString(strings.ToLower(query), " ")

	// Разбиваем на слова
	words := strings.Fields(cleaned)

	// Удаляем стоп-слова и короткие слова
	var cleanWords []string
	for _, word := range words {
		if len(word) > 2 && !r.stopWords[word] {
			cleanWords = append(cleanWords, word)
		}
	}

	return cleanWords
}

func (r *Retriever) calculateScore(doc Document, queryWords []string) float64 {
	score := 0.0
	headerLower := strings.ToLower(doc.Header)
	keywordsLower := strings.ToLower(doc.Keywords)

	for _, word := range queryWords {
		// Поиск точного совпадения в заголовке (больший вес)
		if strings.Contains(headerLower, word) {
			score += 3.0
		}

		// Поиск точного совпадения в ключевых словах
		if strings.Contains(keywordsLower, word) {
			score += 2.0
		}

		// Поиск частичного совпадения в заголовке
		headerWords := strings.Fields(headerLower)
		for _, headerWord := range headerWords {
			if strings.Contains(headerWord, word) || strings.Contains(word, headerWord) {
				score += 1.0
			}
		}

		// Поиск частичного совпадения в ключевых словах
		keywordWords := strings.Fields(keywordsLower)
		for _, keywordWord := range keywordWords {
			if strings.Contains(keywordWord, word) || strings.Contains(word, keywordWord) {
				score += 0.5
			}
		}
	}

	return score
}

func (r *Retriever) Search(query string, topK int) []Document {
	// Очищаем запрос от стоп-слов и знаков препинания
	queryWords := r.cleanQuery(query)
	if len(queryWords) == 0 {
		return []Document{}
	}

	// Вычисляем релевантность для каждого документа
	var scored []DocumentScore
	for _, doc := range r.docs {
		score := r.calculateScore(doc, queryWords)
		if score > 0 {
			scored = append(scored, DocumentScore{
				Document: doc,
				Score:    score,
			})
		}
	}

	// Сортируем по убыванию релевантности
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score > scored[j].Score
	})

	// Возвращаем topK результатов
	var results []Document
	for i, item := range scored {
		if i >= topK {
			break
		}
		results = append(results, item.Document)
	}

	return results
}
