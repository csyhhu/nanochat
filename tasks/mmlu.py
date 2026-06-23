"""
The MMLU dataset.
https://huggingface.co/datasets/cais/mmlu
"""

from datasets import load_dataset
from tasks.common import Task, render_mc

class MMLU(Task):

    letters = ('A', 'B', 'C', 'D')
    groups = ('abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions')

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["all"], f"subset {subset} must be all"
        assert split in ["auxiliary_train", "validation", "dev", "test"], f"split {split} must be auxiliary_train|validation|dev|test"
        self.subset = subset
        self.split = split
        self.ds = load_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"] # the question text
        choices = row["choices"] # the text of each choice
        answer = row["answer"] # index of the answer, e.g. 0,1,2,3 (for A,B,C,D)
        subject = row["subject"] # e.g. "college_biology", "college_chemistry", etc.
        assert len(choices) == 4, "MMLU should have 4 choices"
        # create and return the Conversation object
        user_message = render_mc(question, self.letters, choices)
        assistant_message = self.letters[answer]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        conversation = {
            "messages": messages,
            "subject": subject, # might be useful later for grouping metrics by subject
            "letters": self.letters, # useful during evaluation, so we can narrow and clamp the assistant prediction to one of the letters
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        # Extract the predicted answer letter from the response.
        # The model may output a full paragraph like "The correct answer is A, 6.0 eV."
        # We need to find the first occurrence of A/B/C/D that appears as a standalone letter.
        pred = self._extract_answer_letter(assistant_response)
        if pred is None:
            # If no letter can be extracted, treat as wrong answer
            return False
        assistant_message = conversation['messages'][-1]['content'] # e.g. "A"
        return pred == assistant_message

    def _extract_answer_letter(self, text):
        """Extract the answer letter (A/B/C/D) from a potentially long response."""
        import re
        # Try common answer patterns first
        patterns = [
            r'answer\s+is\s+([A-D])',      # "answer is A"
            r'correct\s+is\s+([A-D])',     # "correct is A"
            r'choose\s+([A-D])',           # "choose A"
            r'option\s+([A-D])',           # "option A"
            r'choice\s+([A-D])',           # "choice A"
            r'([A-D])[\)\.]',              # "A)" or "A."
            r'\b([A-D])\b',                # standalone A, B, C, D
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None

    def reward(self, conversation, assistant_response):
        """Use simple 0-1 reward, same as GSM8K and SpellingBee."""
        is_correct = self.evaluate(conversation, assistant_response)
        return float(is_correct)
