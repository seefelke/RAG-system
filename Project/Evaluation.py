from bert_score import score as bertscore

from ragas import evaluate
from ragas.metrics import AnswerAccuracy
from ragas.llms import LangchainLLMWrapper

def evaluate_bertscore(preds, refs):
    P, R, F1 = bertscore(preds, refs, lang="de", model_type="bert-base-multilingual-cased", rescale_with_baseline=True)
    return {
        "precision" : P,
        "recall" : R,
        "f1" : F1
    }

def evaluate_ragas(llm, dataset, embedding):
    llm = LangchainLLMWrapper(llm)
    answer_accuracy = AnswerAccuracy(llm = llm)
    result = evaluate(
        dataset=dataset,
        metrics=[answer_accuracy],
        llm=llm,
        embeddings=embedding
    )
    return result