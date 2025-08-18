from bert_score import BERTScorer

from ragas import evaluate
from ragas.metrics import AnswerAccuracy, ContextPrecision, Faithfulness, ResponseRelevancy, AnswerCorrectness
from ragas.llms import LangchainLLMWrapper

def evaluate_bertscore(preds, refs):
    scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang='de')
    #scorer = BERTScorer(model_type="google-bert/bert-base-multilingual-cased", lang='de', num_layers=12)
    #P, R, F1 = bertscore(preds, refs, lang=None, model_type="google-bert/bert-base-german-cased", num_layers=None, verbose=True)
    P, R, F1 = scorer.score(preds, refs)
    return {
        "precision" : P,
        "recall" : R,
        "f1" : F1
    }

def evaluate_ragas(llm, dataset, embedding):
    llm = LangchainLLMWrapper(llm)
    answer_accuracy = AnswerAccuracy(llm = llm)
    context_precision = ContextPrecision(llm = llm)
    faithfulness = Faithfulness(llm = llm)
    response_relevancy = ResponseRelevancy(llm = llm)
    answer_correctness = AnswerCorrectness(llm = llm)
    result = evaluate(
        dataset=dataset,
        metrics=[answer_accuracy, context_precision, faithfulness, response_relevancy, answer_correctness],
        llm=llm,
        embeddings=embedding
    )
    return result
