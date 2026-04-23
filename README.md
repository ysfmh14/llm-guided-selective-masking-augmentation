# llm-guided-selective-masking-augmentation

This repository provides the implementation of the **LLM-guided text augmentation framework using selective masking strategies** introduced in the proposed framework:

***Improving Text Classification in the One Health Context through Selective Masking Data Augmentation with LLM-Guided Sampling***

---

## 📄 Abstract

Ensuring efficient One Health surveillance using textual sources presents significant challenges, particularly due to the limited availability of labeled data for specialized classification tasks, such as thematic classification in plant health, syndromic surveillance, and epidemic misinformation detection. To address this issue, this paper proposes a novel data augmentation framework based on selective masking strategies combined with large language model (LLM)-based sampling. we introduce two families of selective masking–based data augmentation strategies: lexical and non-lexical. Each family is implemented in a standard variant (AuSeMa-L-LLM and AuSeMa-NL-LLM), and a TF-IDF-weighted variant (AuSeMa-LT-LLM and AuSeMa-NLT-LLM). These strategies apply controlled perturbations by masking specific tokens and generating context-aware replacements using BERTBase. For LLM-based sampling, we employ Mistral-7B and LLaMA-3.1-8B. To see how these augmentation strategies actually performed, we fine-tuned BERTBase as the classification model using both the new augmented sets and the original data as our baseline. We then ran a series of experiments across different datasets to see how things like masking rates and data distribution affected the results. The results indicate that a masking rate of 20\% strikes the best balance between contextual coherence and lexical diversity. Moreover, the proposed approaches consistently outperform the baseline, as well as traditional augmentation techniques such as back-translation and recent LLM-based reformulation methods. In particular, the AuSeMa-NL-LLM strategy demonstrates strong robustness across the studied datasets, achieving notable improvements while maintaining computational efficiency.
---

---

## 🧩 Project Structure

    llm-guided-selective-masking-augmentation/
    │
    ├── strategies/
    │ ├── AuSeMa-NL-LLM.py
    │ ├── AuSeMa-L-LLM.py
    │ ├── AuSeMa-NLT-LLM.py
    │ └── AuSeMa-LT-LLM.py
    │
    ├── common/
    │ ├── llm.py
    │ ├── bert-loader.py
    │ ├── data-loader.py
    │ ├── lexicon-loader.py
    │
    ├── config.py
    ├── requirements.txt
    └── README.md


---

## 🚀 How to Run
 ### 1. Clone the Repository
    git clone https://github.com/your-username/llm-guided-selective-masking-augmentation.git
    cd llm-guided-selective-masking-augmentation
### 2. Install Dependencies
Make sure you have Python 3.8+ installed, then run:

    pip install -r requirements.txt

### 3. Prepare Data and Parameters
Place your datasets and lexicon files in the appropriate format, and configure all required parameters in config.py.

Update the following variables:

 - ***DATA_PATH:*** path to the main dataset file
 - ***TEXT_COL:*** name of the text column in the dataset
 - ***LABEL_COL:*** name of the label column in the dataset
 - ***LEXICON_PATH:*** path to the domain lexicon file
 - ***LLM_MODEL:*** name of the LLM to use for sampling (e.g., "unsloth/Meta-Llama-3.1-8B-Instruct")
 - ***HF_TOKEN:*** Hugging Face access token for loading the LLM

## 📁 Citation
If you use this code, please cite the original paper:

***Improving Text Classification in the One Health Context through Selective Masking Data Augmentation with LLM-Guided Sampling***
## 📬 Contact
If you have any questions, encounter issues with the code, or would like to know more about our work, please contact the corresponding author:

📧 Personal email (permanent): ysfmh2002@gmail.com

📧 Professional email (not sure if permanent): mahdoubi.youssef@usms.ac.ma
