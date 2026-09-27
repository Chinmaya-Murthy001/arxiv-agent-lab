---
title: arXiv Research Agent
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.28.0
python_version: '3.12'
app_file: app.py
pinned: false
---

# Lab 2: arXiv Research Agent

## Purpose

This project is a Gradio research assistant for finding and reviewing academic papers from arXiv. It retrieves paper records and PDFs from arXiv and uses a Hugging Face language model to summarize or compare selected papers.

## Starting Point

The lab began with a basic Gradio Space template. Its search response contained hard-coded sample papers, and the agent/model path was not connected to the interface. The work started by replacing the sample output with live arXiv search, then expanding the app into a paper reading and analysis workflow.

## Features

- Search arXiv by keywords, with category, author, submission-date, and result-count filters.
- Retrieve arXiv PDFs and extract text from up to 12 pages and 12,000 characters per paper.
- Summarize one paper or compare two papers using the extracted PDF text.
- Show page-number citations with short source excerpts and links to those PDF pages.
- Save papers to a session reading list and export them as a BibTeX file.
- Validate search inputs, selected papers, and PDF URLs; limit PDF size and extraction length.

## How to Use

1. Enter search terms and optionally set filters, then select **Search**.
2. Select a result and choose **Read PDF** or **Summarize**. Select two results to compare them.
3. Save selected results to the reading list and choose **Export BibTeX** to download citations.

## Current Stage

The app is a working prototype. Live arXiv search, filtered queries, PDF extraction, and BibTeX export have been tested. Summary and comparison workflows have been checked with mocked model responses; live model inference still needs a valid Hugging Face token. The reading list is kept only for the current session.

## Run Locally

Install dependencies with `pip install -r requirements.txt`, set `HF_TOKEN` in the environment (or a local `.env` file), then run `python app.py`. `HF_MODEL_ID` can optionally select another supported inference model; the default is `Qwen/Qwen2.5-Coder-32B-Instruct`.
