# requirements
echo  installing requirements
pip install -r requirements.txt

# loading dataset
echo cloning HumanEval dataset from the Hub to load it offline later
git clone https://huggingface.co/datasets/openai_humaneval
