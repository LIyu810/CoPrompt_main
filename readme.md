# CoPrompt

CoPrompt is a contrast-prompt tuning method specifically designed for Chinese dialogue character relationship extraction. This repository contains the source code and datasets utilized in our paper, "CoPrompt: A Contrast-prompt Tuning Method for Chinese Dialogue Character Relationship Extraction."

## Requirements

To install the requirements, please use the following pip command. It will install all the necessary libraries and packages that are mentioned in the `requirements.txt` file.

```
pip install -r requirements.txt
```

## Running the Experiments

You can follow the steps below to run the experiments detailed in our paper:

### 1. Generate the Label Words

Firstly, use the script `get_label_word.py` to generate the label words that will be used during the training process. You can run this script using the following command:

```
python get_label_word.py
```

### 2. Execute the Main Script

After generating the label words, you can start the main experiment by executing the `chinese.sh` script in the `scripts` directory. Run the following command in your terminal:

```
bash scripts/chinese.sh
```

## Acknowledgements

We would like to express our gratitude to the creators of KnowPrompt: "Knowledge-aware Prompt-tuning with Synergistic Optimization for Relation Extraction." We have borrowed part of our code from their work, and their contributions have significantly aided our research.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://chat.openai.com/LICENSE.md) file for details

## Contributing

We welcome any contributions to the CoPrompt project. Please feel free to submit pull requests or raise issues. Our team will review them promptly.

## Contact

If you have any questions or need further clarification, please do not hesitate to reach out to us. You can find our contact information in the authors' section of our paper.









[comment]: <> (### CoPrompt)

[comment]: <> (Code and datasets for paper CoPrompt: A Contrast-prompt Tuning Method for Chinese Dialogue Character Relationship Extraction)

[comment]: <> (### Requirements)

[comment]: <> (```)

[comment]: <> (pip install -r requirements.txt)

[comment]: <> (```)

[comment]: <> (### Run the experiments)

[comment]: <> (#### 1.Use the comand below to get the answer words to use in the training.)

[comment]: <> (```)

[comment]: <> (python get_label_word.py)

[comment]: <> (```)

[comment]: <> (#### 2.Let's run)

[comment]: <> (```)

[comment]: <> (>> bash scripts/dialogue.sh)

[comment]: <> (```)

[comment]: <> (### Acknowledgement)

[comment]: <> (Part of our code is borrowed from code of KnowPrompt: Knowledge-aware Prompt-tuning with Synergistic Optimization for Relation Extraction, many thanks.)