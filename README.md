# Master Thesis RAG System

## System Overview

This repository consists of 2 major parts: The backend RAG system and the frontend interface. The direct use case for this project was the utilization of RAG to feed information about a local museum exhibition to a LLM to provide museum visitors with a way to directly ask about things they see in front of them.
The frontend interface was developed as an example on how to couple the backend system to a frontend interface and to provide an evaluation interface for the thesis. The following illustration provides a high level overview of the system:
![System Overview](Git%20images/system%20overview.PNG)

## Backend

The backend divides a provided input PDF into chunks and saves them to a vector store of choice. A user query from the provided frontend interface is combined with the retrieved context from the RAG system and a system prompt to be fed into a LLM of choice. Additionally, the final result can be evaluated with the optional evaluation backend. An exemplary use for this can be seen in the frontend.
## Frontend

![Frontend](Git%20images/frontend.PNG)
