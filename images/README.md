# Blog Images

This directory contains images used in the blog post `rag_chatbot_blog.md`.

## Required Images

### 1. `hr_chatbot_ui.png`
**Description**: Screenshot of the HR Chatbot UI showing a conversation
- Shows the Streamlit interface with chat history
- Displays a conversation about notice period at grade 4
- User: "Hi I am Kanav" and "What is the notice period at grade 4?"
- Chatbot response with citations [1, 2, 4]

**Usage**: Used in the "Real-World Example: HR Policy Query" section

### 2. `evaluation_scores_chart.png`
**Description**: Bar chart showing evaluation metrics
- Title: "Feedback"
- Metrics displayed:
  - correctness: ~0.78 (purple bar)
  - groundedness: 1.0 (light purple bar) - perfect score
  - relevance: ~0.97 (blue bar)
  - retrieval_relevance: ~0.95 (light blue bar)
  - scannability: ~0.78 (dark blue bar)
- X-axis: "#1: hr-ch..." (evaluation run identifier)
- Y-axis: Scores from 0 to 1

**Usage**: Used in the "Performance at Scale" section

## Adding Images

1. Save your screenshots/charts with the exact filenames listed above
2. Place them in this `images/` directory
3. The blog post references them using relative paths: `images/hr_chatbot_ui.png` and `images/evaluation_scores_chart.png`

## Image Specifications (for Medium)

- **Recommended width**: 1200px for Medium articles
- **Format**: PNG or JPG
- **File size**: Optimize for web (under 500KB if possible)
- **Alt text**: Already included in the blog post markdown
