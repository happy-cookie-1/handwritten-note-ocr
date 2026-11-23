# Development log

## 17 November 2026

The recognizer now transcribes entire handwritten pages. Most words in the samples I tested were recognized correctly. I estimated the character error rate at around 5%, with extra or incorrect punctuation accounting for much of the error. This was an observation from my development samples. I did not record a test-set size or a strict versus normalized score alongside this estimate.

I want to test whether trimming the margins of each line improves recognition. The intended place for this step is at the end of the per-line crop processing in `split_lines()`. The current `trim_line_margins()` function is a placeholder.

## 18 November 2026

I integrated Qwen3 4B through Ollama to analyze a page after transcription. It generates a title, subject and keywords from the recognized text. The page workflow writes those fields at the beginning of the output text file, followed by the transcription.

## 19 November 2026

I updated `flag_tall_candidates()` to adapt its minimum height to the handwriting on each page. An ink group qualifies for assessment if it has at least 40 original ink pixels and is at least half the median height of all candidates. For eligible groups, the function uses nearby groups to estimate the expected height, falling back to the page-wide median of qualifying groups when too few neighbors are available. It flags a possible merge when the height exceeds 1.6 times the expected height.

I also started documenting what each function does and what it returns.
