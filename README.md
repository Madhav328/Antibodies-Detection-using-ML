Antibody Fitness Prediction & Developability Analysis Using Machine Learning

Project Overview

This project builds a complete computational pipeline to analyze antibody sequences, reconstruct heavy chains, extract meaningful features using the ESM-2 transformer model, and predict antibody fitness and developability using multiple machine learning models.

The work satisfies all expectations outlined in the Course-Based Design Project criteria, including novelty, data selection, descriptive analytics, predictive modeling, prescriptive analysis, results, and comparative evaluation.

⸻

Objectives
	•	Reconstruct antibody sequences by filling masked CDR3 regions.
	•	Represent each antibody using ESM-2 protein embeddings.
	•	Perform descriptive analytics on CDR3 diversity and biochemical properties.
	•	Predict continuous fitness values using regression models.
	•	Classify variants as winner vs loser using classification models.
	•	Apply developability filters to shortlist viable antibody candidates.
	•	Compare multiple ML models and justify the best-performing one.

⸻

Dataset Description

The dataset consists of 13,975 antibody variants.
Each variant includes:
	•	A masked heavy-chain sequence.
	•	A winning CDRH3 candidate.
	•	A losing CDRH3 candidate.
	•	Experimental “winning” and “losing” fitness values.
	•	A complete paired light-chain sequence.

The dataset required BOM removal, delimiter auto-detection, and integrity checks. Masked heavy chains were reconstructed by substituting the correct winning CDRH3 segment.

⸻

Feature Engineering

Two categories of features were extracted:

1. ESM-2 Transformer Embeddings

These embeddings capture structural, biochemical, and evolutionary features from the sequence alone.
Embeddings were computed for:
	•	The reconstructed heavy chain.
	•	The paired light chain.
	•	The winning CDR3 loop.

All three embeddings were concatenated into a single feature vector for machine-learning tasks.

2. Biophysical Properties

Computed using sequence-analysis tools:
	•	CDR3 length
	•	pI (isoelectric point)
	•	Instability index
	•	GRAVY hydropathy score

These properties helped evaluate protein stability and solubility, essential for therapeutic developability.

⸻

Descriptive Analytics

The descriptive analysis included:
	•	CDR3 length distribution, showing clear multi-modal peaks typical of natural immune repertoires.
	•	Hydropathy vs instability analysis, indicating that most sequences lie within stable and soluble biochemical space.
	•	PCA of ESM embeddings, revealing clustering patterns and separability tendencies between variants.
	•	t-SNE visualization, demonstrating a structured manifold rather than random embedding behavior.

These analyses provided important insights into antibody diversity and molecular behavior.

⸻

Predictive Modeling

Several machine-learning algorithms were evaluated on the fitness prediction task:
	•	Random Forest Regressor
	•	Extra Trees Regressor
	•	Gradient Boosting Regressor
	•	Support Vector Regressor
	•	Linear Regression
	•	KNN (k=5)

Models were evaluated using:
	•	Mean Squared Error
	•	R² Score
	•	Spearman Correlation
	•	Classification Accuracy
	•	ROC AUC

Best Predictor: Random Forest

Random Forest achieved strong balanced performance across all metrics, including:
	•	High R²
	•	Low MSE
	•	Excellent Spearman correlation
	•	Stable behavior and resistance to noise

A composite score combining MSE, R², and Spearman confirmed Random Forest as the best overall model.

⸻

Classification Task

Winner vs loser classification was based on fitness comparison.
Although accuracy was high due to class imbalance, ROC AUC reflected moderate discriminative power.
Regression proved to be more informative and reliable.

⸻

Prescriptive Analytics

The project combined predicted fitness scores with developability properties such as:
	•	Instability index threshold
	•	GRAVY hydropathy cutoff

This allowed prioritizing antibody sequences that were not only high-fitness but also biochemically stable and manufacturable.

This step provides real-world value by identifying the most promising candidates for experimental validation.

⸻

Overall Workflow Summary
	1.	Load and clean dataset.
	2.	Reconstruct masked heavy-chain sequences.
	3.	Extract ESM embeddings and biophysical features.
	4.	Perform descriptive analytics.
	5.	Train and compare ML models for fitness prediction.
	6.	Classify variants based on fitness differences.
	7.	Evaluate all models visually and numerically.
	8.	Apply developability filters to recommend top variants.
	9.	Present final model justification with composite scoring.

⸻

Learning Outcomes

Technical Learnings
	•	Learned how to extract ESM-2 embeddings for protein sequences.
	•	Built and optimized ML models using Python under CPU-only hardware constraints.
	•	Understood key metrics used in regression and classification.
	•	Developed skills in data cleaning, visualization, and model comparison.

Domain Learnings
	•	Gained deeper understanding of antibody structure, especially the role of CDR3.
	•	Learned how developability metrics influence therapeutic viability.
	•	Understood how ML pipelines are integrated into antibody engineering workflows.

Project & Communication Skills
	•	Structured an end-to-end scientific workflow.
	•	Debugged complex errors (BOM, delimiter mismatch, sklearn compatibility).
	•	Designed clear visualizations and presentation-ready plots.
	•	Learned to justify model choices scientifically.

⸻

Conclusion

The project successfully demonstrates how machine learning, guided by transformer-based protein embeddings, can accurately predict antibody fitness and evaluate developability characteristics.

Random Forest emerged as the most robust and balanced model, and the final pipeline provides actionable insights suitable for early-stage antibody optimization and screening.
