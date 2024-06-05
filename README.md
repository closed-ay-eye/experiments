# Recipe App Design Document

## Project Overview

Our project involves developing an app that identifies ingredients from photos and proposes recipes with illustrated steps and narration. This app will leverage various AI technologies to provide an intuitive and helpful cooking assistant.

## Objectives

- Identify ingredients from user-uploaded photos.
- Retrieve relevant recipes based on identified ingredients.
- Generate illustrated steps for the recipes.
- Provide narration for each step of the recipe.

## Technologies and Tools

- **Image Recognition:** Gemini Pro Vision
- **Data Storage and Retrieval:** FAISS, LangChain
- **Recipe Dataset:** [Food.com Recipes with Search Terms and Tags](https://www.kaggle.com/datasets/shuyangli94/foodcom-recipes-with-search-terms-and-tags)
- **Illustration Generation:** DALL-E 3 in ChatGPT
- **Text-to-Speech:** [Text-to-speech model]

## Implementation Plan

### 1. Ingredient Identification

- **Tool:** Gemini Pro Vision
- **Process:** 
  - Users upload a photo of ingredients.
  - The app uses Gemini Pro Vision to analyze the photo and identify the ingredients.

### 2. Recipe Retrieval

- **Tools:** FAISS, LangChain, ChatGPT
- **Dataset:** [Food.com Recipes with Search Terms and Tags](https://www.kaggle.com/datasets/shuyangli94/foodcom-recipes-with-search-terms-and-tags)
- **Process:**
  - Identified ingredients are used to query the Kaggle dataset using FAISS and LangChain.
  - The RAG (Retrieval-Augmented Generation) system queries ChatGPT to return a suitable recipe based on the ingredients.

### 3. Illustration Generation

- **Tool:** DALL-E 3 in ChatGPT
- **Process:**
  - For each step of the recipe, DALL-E 3 generates an illustration to visually guide the user.

### 4. Narration

- **Tool:** Text-to-Speech Model
- **Process:**
  - Text for each recipe step is converted into speech using the text-to-speech model.
  - Users can listen to the instructions while cooking.

## User Interface Design

### Main Features

- **Upload Photo:** Button to upload a photo of ingredients.
- **Ingredient List:** Display the list of identified ingredients.
- **Recipe Display:** Show the retrieved recipe with steps and illustrations.
- **Narration:** Play button for each step to hear the instructions.

### User Flow

1. **Upload Photo:** User uploads a photo of ingredients.
2. **Ingredient Identification:** App processes the photo and displays the identified ingredients.
3. **Recipe Retrieval:** User confirms ingredients and retrieves a recipe.
4. **Illustrated Steps:** User views illustrated steps for the recipe.
5. **Narration:** User listens to the narration for each step.

## Technical Architecture

### Backend

- **Image Processing:** Gemini Pro Vision
- **Data Storage:** FAISS index of the Kaggle dataset
- **Query System:** LangChain with ChatGPT for RAG
- **Illustration Generation:** DALL-E 3
- **Text-to-Speech:** [Text-to-speech model]

### Frontend

- **Framework:** [framework]
- **Components:**
  - Upload Photo Component
  - Ingredient List Component
  - Recipe Display Component
  - Illustration Display Component
  - Narration Component



---

This design document outlines the structure and plan for developing the recipe app. Adjustments and iterations are expected as the project progresses.
