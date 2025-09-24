from bert_score import BERTScorer

from ragas import evaluate
from ragas.metrics import AnswerAccuracy, ContextPrecision, Faithfulness, ResponseRelevancy, AnswerCorrectness
from ragas.llms import LangchainLLMWrapper
import os
import pandas as pd

pd.set_option("display.max_columns", None)

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


def evaluate_ragas_baseline(llm, dataset, embedding):
    llm = LangchainLLMWrapper(llm)
    answer_accuracy = AnswerAccuracy(llm = llm)
    response_relevancy = ResponseRelevancy(llm = llm)
    answer_correctness = AnswerCorrectness(llm = llm)
    result = evaluate(
        dataset=dataset,
        metrics=[answer_accuracy, response_relevancy, answer_correctness],
        llm=llm,
        embeddings=embedding
    )
    return result


def evaluate_data(run_dir):
    # collect all CSV files for a given parent folder name like mistral - FAISS
    run_dir = os.path.join("logs", run_dir)
    csv_list = []
    for date in os.listdir(run_dir):
        date_path = os.path.join(run_dir, date)
        if not os.path.isdir(date_path):
            continue

        for timestamp in os.listdir(date_path):
            ts_path = os.path.join(date_path, timestamp)
            if not os.path.isdir(ts_path):
                continue

            # look for CSVs inside this timestamp folder
            for file in os.listdir(ts_path):
                if file.endswith(".csv"):
                    csv_path = os.path.join(ts_path, file)
                    try:
                        csv_list.append(pd.read_csv(csv_path))
                    except Exception as e:
                        print(f"Skipping {csv_path}: {e}")


class RAGEvaluator:
    def __init__(self, base_folder: str, skip_size = 0, skip_amount = 10):
        """
        base_folder = path to a model+vectorstore folder
        Example: logs/mistral_faiss
        skip_size: skip all chunks with less than this size
        skip_amount: skip all chunks with more than this amount
        """
        self.base_folder = base_folder
        self.csv_files = self._collect_csv_files()
        self.dataframes = self._load_dataframes(skip_size, skip_amount)

    def _collect_csv_files(self):
        csvs = []
        for root, _, files in os.walk(self.base_folder):
            for file in files:
                if file.endswith(".csv"):
                    csvs.append(os.path.join(root, file))
        return csvs

    def _load_dataframes(self, skip_size, skip_amount):
        dfs = {}
        for path in self.csv_files:
            try:
                df = pd.read_csv(path)

                chunk_size = int(df.iloc[5, 1])
                chunk_amount = int(df.iloc[7, 1])
                print(chunk_amount)
                # Apply filtering
                if chunk_size < skip_size or chunk_amount > skip_amount:
                    print(f"Skipping {path} due to chunking constraints")
                    continue
                # Only keep rows with real metric values
                df = df[df["Category"] != "Average"]
                dfs[path] = df
            except Exception as e:
                print(f"Skipping {path}: {e}")
        return dfs

    def summarize_metrics(self):
        results = {}
        metric_cols = [
            "Time", "BERTScore_Precision", "BERTScore_Recall", "BERTScore_F1",
            "RAGAS_NvAccuracy", "RAGAS_ContextPrecision", "RAGAS_Faithfulness",
            "RAGAS_AnswerRelevancy", "RAGAS_AnswerCorrectness"
        ]
        for file, df in self.dataframes.items():
            for col in metric_cols:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "."),
                    errors="coerce"
                )
            self.dataframes[file] = df
        combined = []
        for file, df in self.dataframes.items():
            avg_row = df[metric_cols].mean()
            avg_row["file"] = file
            combined.append(avg_row)

        all_df = pd.DataFrame(combined)

        for metric in metric_cols:
            max_idx = all_df[metric].idxmax()
            min_idx = all_df[metric].idxmin()

            abs_max_val = None
            abs_max_file, abs_max_q = None, None
            abs_min_val = None
            abs_min_file, abs_min_q = None, None

            for file, df in self.dataframes.items():
                if metric in df.columns:
                    idxmax = df[metric].idxmax()
                    idxmin = df[metric].idxmin()
                    if not pd.notna(idxmax):
                        print(df[metric])
                        print(file)
                    if pd.notna(df.loc[idxmax, metric]):
                        val = df.loc[idxmax, metric]
                        if abs_max_val is None or val > abs_max_val:
                            abs_max_val = val
                            abs_max_file = file
                            abs_max_q = df.loc[idxmax].get("Question", "")
                    if pd.notna(df.loc[idxmin, metric]):
                        val = df.loc[idxmin, metric]
                        if abs_min_val is None or val < abs_min_val:
                            abs_min_val = val
                            abs_min_file = file
                            abs_min_q = df.loc[idxmin].get("Question", "")

            results[metric] = {
                "highest_avg": float(all_df.loc[max_idx, metric]),
                "highest_avg_file": all_df.loc[max_idx, "file"],
                "lowest_avg": float(all_df.loc[min_idx, metric]),
                "lowest_avg_file": all_df.loc[min_idx, "file"],
                "average": float(all_df[metric].mean()),
                "absolute_highest": float(abs_max_val) if abs_max_val is not None else None,
                "absolute_highest_file": abs_max_file,
                "absolute_highest_question": abs_max_q,
                "absolute_lowest": float(abs_min_val) if abs_min_val is not None else None,
                "absolute_lowest_file": abs_min_file,
                "absolute_lowest_question": abs_min_q,
            }

        return results

    def best_overall_runs(self):
            metric_cols = [
                "BERTScore_Precision", "BERTScore_Recall", "BERTScore_F1",
                "RAGAS_NvAccuracy", "RAGAS_ContextPrecision", "RAGAS_Faithfulness",
                "RAGAS_AnswerRelevancy", "RAGAS_AnswerCorrectness"
            ]

            combined = []
            for file, df in self.dataframes.items():
                avg_val = df[metric_cols].mean().mean()
                combined.append({"file": file, "overall_avg": avg_val})

            df_all = pd.DataFrame(combined)
            best = df_all.loc[df_all["overall_avg"].idxmax()]
            return {"best_file": best["file"], "avg_score": float(best["overall_avg"])}

    def best_query_performance(self):
        metric_cols = [
            "BERTScore_Precision", "BERTScore_Recall", "BERTScore_F1",
            "RAGAS_NvAccuracy", "RAGAS_ContextPrecision", "RAGAS_Faithfulness",
            "RAGAS_AnswerRelevancy", "RAGAS_AnswerCorrectness"
        ]
        best_queries = []
        for file, df in self.dataframes.items():
            if "Category" in df.columns and "Average" in df["Category"].values:
                df = df[df["Category"] != "Average"]
            for _, row in df.iterrows():
                if not set(metric_cols).issubset(row.index):
                    continue
                avg = row[metric_cols].mean()
                best_queries.append({
                    "file": file,
                    "question": row.get("Question", ""),
                    "avg_score": avg
                })
        best_df = pd.DataFrame(best_queries)
        best_row = best_df.loc[best_df["avg_score"].idxmax()]
        return {
            "best_file": best_row["file"],
            "question": best_row["question"],
            "avg_score": float(best_row["avg_score"])
        }


    def per_question_averages(self):
        metric_cols = [
            "BERTScore_Precision", "BERTScore_Recall", "BERTScore_F1",
            "RAGAS_NvAccuracy", "RAGAS_ContextPrecision", "RAGAS_Faithfulness",
            "RAGAS_AnswerRelevancy", "RAGAS_AnswerCorrectness"
        ]

        all_rows = []
        for file, df in self.dataframes.items():
            all_rows.append(df[["Question"] + metric_cols].dropna())
        merged = pd.concat(all_rows)
        grouped = merged.groupby("Question", sort=False)[metric_cols].mean()
        grouped["Average"] = grouped.mean(axis=1)
        return grouped

    def per_level_averages(self):
        metric_cols = [
            "BERTScore_Precision", "BERTScore_Recall", "BERTScore_F1",
            "RAGAS_NvAccuracy", "RAGAS_ContextPrecision", "RAGAS_Faithfulness",
            "RAGAS_AnswerRelevancy", "RAGAS_AnswerCorrectness"
        ]

        all_rows = []
        for file, df in self.dataframes.items():
            # keep only rows with a Level value
            all_rows.append(df[["Level"] + metric_cols].dropna(subset=["Level"]))

        merged = pd.concat(all_rows)

        # group by Level ("leicht", "schwer", etc.) and average the metrics
        grouped = merged.groupby("Level", sort=False)[metric_cols].mean()

        # add an "Average" column = mean across all metrics per Level
        grouped["Average"] = grouped.mean(axis=1)

        return grouped
