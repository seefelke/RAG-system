import unittest
import Extraction
import json
import random
import datetime
import Evaluation
import Chatting
from datasets import Dataset
import config
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

class RetrievalTesting(unittest.TestCase):

    def setUp(self):
        self.vectorstore = Extraction.get_vectorstore()
        self.retriever = self.vectorstore.as_retriever()

    def run_similarity_score(self, query):
        results = self.vectorstore.similarity_search_with_score(query)
        doc, score = results[0]
        return score

    def run_similarity_score_min(self, query, min_score):
        score = self.run_similarity_score(query)
        self.assertTrue(score >= min_score, "Score is less than min_score.")

    def run_lexical_search(self, query, keywords):
        results = self.retriever.invoke(query)
        contents = [doc.page_content for doc in results]
        for word in keywords:
            self.assertTrue(any(word in c.lower() for c in contents),
                            "Lexical search failed to find exact match.")

    def test_similarity_score(self):
        self.run_similarity_score_min("Welche Überlebensstrategien verwenden Tiere?", 0.1)

    def test_lexical_search(self):
        self.run_lexical_search("Welche Überlebensstrategien verwenden Tiere?", {"mimese", "mimikry"})

    def test_rag_system(self):
        """ Tests the RAG system with 2 questions from each category and saves the results to a csv file."""
        if not os.path.exists("questions_and_answers.json"):
            raise FileNotFoundError(f"Question/answer file not found at path: {"questions_and_answers.json"}")

        with open("questions_and_answers.json", "r", encoding="utf-8") as f:
            questions_and_answers = json.load(f)
            today = datetime.date.today().isoformat()
            time_str = datetime.datetime.now().strftime("%H-%M")
            model_str = config.MODEL_NAME + " " + config.STORE_TYPE
            log_dir = os.path.join("logs", model_str, today, time_str)
            with tqdm(total=4*3*4) as pbar:
                for i in range(1,config.CHUNK_SIZE_STEPS):
                    for j in range(1,config.CHUNK_OVERLAP_STEPS):
                        for k in range(1,config.CHUNK_AMOUNT_STEPS):
                            # Prepare logs directory
                            run_number = (i-1)*4*3 + (j-1)*4 + k
                            print("New run start: " + str(run_number))
                            config.CHUNK_SIZE = config.CHUNK_SIZE_STEP_AMOUNT * i
                            config.CHUNK_OVERLAP = round(config.CHUNK_OVERLAP_STEP_AMOUNT * config.CHUNK_SIZE)
                            config.CHUNK_AMOUNT = k
                            Chatting.setup_vectorbase()
                            Chatting.setup_chatbot()
                            qa_chain = Chatting.retrieval_chain
                            #qa_chain = Chatting.qa_chain
                            #memory = Chatting.memory
                            today = datetime.date.today().isoformat()
                            os.makedirs(log_dir, exist_ok=True)
                            random.seed(123)
                            embedding = Extraction.embedding

                            csv_path = os.path.join(log_dir, f"rag_eval_{today}_{time_str}_{run_number}.csv")
                            categories = ['einfache_fragen', 'schwere_fragen'] # 'spezielle_fragen'
                            all_preds = []
                            all_refs = []
                            ragas_dataset = []
                            results = []

                            metadata = {
                                "Test ID": run_number,
                                "Date": today,
                                "Chatbot model": config.MODEL_NAME,
                                "Embedding model": config.EMBEDDINGS,
                                "Vectorstore": config.STORE_TYPE,
                                "Chunking size": config.CHUNK_SIZE,
                                "Chunking overlap": config.CHUNK_OVERLAP,
                                "Chunk amount": config.CHUNK_AMOUNT,
                                "Additional notes": config.EXTRA_NOTES,
                            }

                            for category in categories:
                                questions = questions_and_answers.get(category, [])
                                samples = questions[0:5]
                                for item in samples:
                                    question = item['frage']
                                    reference = item['antwort']
                                    start_time = time.time()
                                    result = qa_chain.invoke({
                                        "input": question
                                    })
                                    #result = qa_chain.invoke({"question": question,
                                     #                         "chat_history": memory.chat_memory.messages})
                                    run_time = time.time() - start_time
                                    prediction = result.get('answer', '') if isinstance(result, dict) else str(result)
                                    sources = result.get('context', [])
                                    contexts = [doc.page_content for doc in sources]
                                    all_preds.append(prediction)
                                    all_refs.append(reference)

                                    ragas_dataset.append({
                                        "question": question,
                                        "ground_truth": reference,
                                        "answer": prediction,
                                        "contexts": contexts
                                    })

                                    results.append({
                                        "Category": category,
                                        "Level": "",  # not applicable here
                                        "Question": question,
                                        "Ground Truth": reference,
                                        "Prediction": prediction,
                                        "Context": contexts,
                                        "Time": run_time
                                    })

                            fragenpaare = questions_and_answers.get("fragenpaare", [])
                            fragenpaare_samples = fragenpaare[0:5]

                            for pair in fragenpaare_samples:
                                for level in ["leicht", "schwer"]:
                                    question = pair[level]
                                    reference = pair["antwort"]
                                    start_time = time.time()
                                    result = qa_chain.invoke({
                                       "input": question
                                    })
                                    run_time = time.time() - start_time
                                    prediction = result.get('answer', '') if isinstance(result, dict) else str(result)
                                    sources = result.get('context', [])
                                    contexts = [doc.page_content for doc in sources]
                                    all_preds.append(prediction)
                                    all_refs.append(reference)

                                    ragas_dataset.append({
                                        "question": question,
                                        "ground_truth": reference,
                                        "answer": prediction,
                                        "contexts": contexts
                                    })

                                    results.append({
                                        "Category": "fragenpaare",
                                        "Level": level,
                                        "Question": question,
                                        "Ground Truth": reference,
                                        "Prediction": prediction,
                                        "Context": contexts,
                                        "Time": run_time
                                    })

                            ragas_dataset = Dataset.from_list(ragas_dataset)

                            # Evaluate with full set
                            bertscore_result = Evaluation.evaluate_bertscore(all_preds, all_refs)
                            ragas_result = Evaluation.evaluate_ragas(Chatting.eval_llm, ragas_dataset, embedding)
                            ragas_df = ragas_result.to_pandas()
                            # Filter out columns that are non-metrics (e.g. strings)
                            ragas_score_columns = ragas_df.select_dtypes(include=["number"]).columns

                            precisions = bertscore_result["precision"].tolist()
                            recalls = bertscore_result["recall"].tolist()
                            f1s = bertscore_result["f1"].tolist()

                            for n, r in enumerate(results):
                                # BERTScore per row
                                r["BERTScore_Precision"] = round(precisions[n], 5)
                                r["BERTScore_Recall"] = round(recalls[n], 5)
                                r["BERTScore_F1"] = round(f1s[n], 5)

                                # RAGAS scores per row
                                for column in ragas_score_columns:
                                    r[f"RAGAS_{column.replace('_', ' ').title().replace(' ', '')}"] = round(ragas_df.iloc[n][column], 5)

                            df = pd.DataFrame(results)
                            avg_row = df.select_dtypes(include='number').mean().to_dict()

                            for col in df.columns:
                                if col == "Category":
                                    avg_row[col] = "Average"
                                elif col not in avg_row:
                                    avg_row[col] = "-"

                            avg_df = pd.DataFrame([avg_row])
                            df = pd.concat([df, avg_df], ignore_index=True)
                            metric_cols = [
                                "BERTScore_Precision", "BERTScore_Recall", "BERTScore_F1",
                                "RAGAS_NvAccuracy", "RAGAS_ContextPrecision", "RAGAS_Faithfulness",
                                "RAGAS_AnswerRelevancy", "RAGAS_AnswerCorrectness"
                            ]
                            # Insert metadata rows at the top
                            meta_rows = [{"Category": f"# {key}", "Question": str(value)} for key, value in metadata.items()]
                            meta_df = pd.DataFrame(meta_rows)
                            final_df = pd.concat([meta_df, pd.DataFrame([{}]), df], ignore_index=True)
                            final_df[metric_cols] = final_df[metric_cols].astype(float)
                            # Save to CSV
                            final_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                            pbar.update(1)
                            print(f" Evaluation CSV saved to: {csv_path}")

    def update_average_graph(self):
        N = 20
        # Collect all CSV paths
        csv_files = []
        for root, dirs, files in os.walk("logs"):
            for file in files:
                if file.endswith(".csv"):
                    path = os.path.join(root, file)
                    mod_time = os.path.getmtime(path)
                    csv_files.append((mod_time, path))

        csv_files = sorted(csv_files, key=lambda x: x[0], reverse=True)[:N]

        # Extract average rows
        avg_rows = []
        timestamps = []

        for _, path in reversed(csv_files):
            try:
                df = pd.read_csv(path)
                avg_row = df[df.iloc[:, 0] == "Average"]
                if not avg_row.empty:
                    avg_row = avg_row.select_dtypes(include="number")
                    avg_row["log_file"] = os.path.basename(path)
                    avg_rows.append(avg_row)
                    timestamps.append(
                        datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M"))
            except Exception as e:
                print(f"Skipping {path}: {e}")

        if avg_rows:
            avg_df = pd.concat(avg_rows, ignore_index=True)
            avg_df["Timestamp"] = timestamps

            avg_df.set_index("Timestamp", inplace=True)
            avg_df.drop(columns=["log_file"], inplace=True, errors="ignore")

            # Plot
            plt.figure(figsize=(12, 6))
            avg_df.plot(marker='o', figsize=(14, 7))
            plt.title(f"Average Metrics over Last {len(avg_df)} Runs")
            plt.xlabel("Run Timestamp")
            plt.ylabel("Score")
            plt.ylim(0, 1.05)
            plt.grid(True)
            plt.legend(loc="best")
            plt.tight_layout()

            graph_path = os.path.join("logs", "summary_graph.png")
            plt.savefig(graph_path)
            print("Graph saved to {graph_path}")
            plt.close()


# A bit of duplicated code, but I want this to be a separate function
def generate_groundtruth(runs):
    if not os.path.exists("questions_and_answers.json"):
        raise FileNotFoundError("questions_and_answers.json not found!")

    with open("questions_and_answers.json", "r", encoding="utf-8") as f:
        questions_and_answers = json.load(f)

    today = datetime.date.today().isoformat()
    time_str = datetime.datetime.now().strftime("%H-%M")
    model_str = config.MODEL_NAME + "_vanilla"
    log_dir = os.path.join("logs", model_str, today, time_str)
    os.makedirs(log_dir, exist_ok=True)

    with tqdm(total=runs * (2 * 5 + 5 * 2)) as pbar:
        # categories = 2x5 questions, fragenpaare = 5x2 questions
        all_preds = []
        all_refs = []
        ragas_dataset = []
        results = []

        categories = ["spezielle_fragen"]#['einfache_fragen', 'schwere_fragen']
        for category in categories:
            questions = questions_and_answers.get(category, [])
            samples = questions[0:5]

            for n in range(runs):

                for item in samples:
                    question = item['frage']
                    reference = item['antwort']
                    start_time = time.time()
                    result = Chatting.llm.invoke(question)
                    run_time = time.time() - start_time

                    prediction = result if isinstance(result, str) else getattr(result, "content", str(result))

                    all_preds.append(prediction)
                    all_refs.append(reference)

                    results.append({
                        "Category": category,
                        "Level": "",
                        "Run": n + 1,
                        "Question": question,
                        "Ground Truth": reference,
                        "Prediction": prediction,
                        "Context": "",
                        "Time": run_time,
                    })

                    ragas_dataset.append({
                        "question": question,
                        "ground_truth": reference,
                        "answer": prediction,
                        "contexts": [""]
                    })

                    pbar.update(1)

        fragenpaare = questions_and_answers.get("fragenpaare", [])
        fragenpaare_samples = fragenpaare[0:5]
        for n in range(runs):
            for pair in fragenpaare_samples:
                for level in ["leicht", "schwer"]:
                    question = pair[level]
                    reference = pair["antwort"]

                    start_time = time.time()
                    result = Chatting.llm.invoke(question)
                    run_time = time.time() - start_time

                    prediction = result if isinstance(result, str) else getattr(result, "content", str(result))
                    all_preds.append(prediction)
                    all_refs.append(reference)
                    results.append({
                        "Category": "fragenpaare",
                        "Level": level,
                        "Run": n + 1,
                        "Question": question,
                        "Ground Truth": reference,
                        "Prediction": prediction,
                        "Context": "",
                        "Time": run_time,
                    })

                    ragas_dataset.append({
                        "question": question,
                        "ground_truth": reference,
                        "answer": prediction,
                        "contexts": [""]
                    })
                    pbar.update(1)

        ragas_dataset = Dataset.from_list(ragas_dataset)

        # Evaluate with full set
        bertscore_result = Evaluation.evaluate_bertscore(all_preds, all_refs)
        ragas_result = Evaluation.evaluate_ragas_baseline(Chatting.eval_llm, ragas_dataset, Extraction.embedding)
        ragas_df = ragas_result.to_pandas()
        ragas_df["RAGAS_Faithfulness"] = 0.0
        ragas_df["RAGAS_ContextPrecision"] = 0.0
        # Filter out columns that are non-metrics (e.g. strings)
        ragas_score_columns = ragas_df.select_dtypes(include=["number"]).columns

        precisions = bertscore_result["precision"].tolist()
        recalls = bertscore_result["recall"].tolist()
        f1s = bertscore_result["f1"].tolist()

        for n, r in enumerate(results):
            # BERTScore per row
            r["BERTScore_Precision"] = round(precisions[n], 5)
            r["BERTScore_Recall"] = round(recalls[n], 5)
            r["BERTScore_F1"] = round(f1s[n], 5)

            # RAGAS scores per row
            for column in ragas_score_columns:
                r[f"RAGAS_{column.replace('_', ' ').title().replace(' ', '')}"] = round(ragas_df.iloc[n][column], 5)

        df = pd.DataFrame(results)
        avg_row = df.select_dtypes(include='number').mean().to_dict()

        for col in df.columns:
            if col == "Category":
                avg_row[col] = "Average"
            elif col not in avg_row:
                avg_row[col] = "-"

        avg_df = pd.DataFrame([avg_row])
        final_df = pd.concat([df, avg_df], ignore_index=True)


        csv_path = os.path.join(log_dir, f"vanilla_eval_{today}_{time_str}.csv")
        final_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Ground truth generations saved to: {csv_path}")