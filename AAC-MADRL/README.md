# AAC-MADRL: Actor-Attention-Critic Multi-Agent DRL

This repository provides training and deployment scripts for three controllers within the **CityLearn** environment:

- **AAC-MADRL** — *Actor-Attention-Critic Multi-Agent DRL*: a novel attention-based multi-agent actor–critic method for district-scale demand-side management (DSM).
- **SAC**
- **MARLISA**

The code is associated with the paper:

> **S. Savino, T. Minella, Z. Nagy, A. Capozzoli (2025)**  
> *A scalable demand-side energy management control strategy for large residential districts based on an attention-driven multi-agent DRL approach*, **Applied Energy**.  
> See the **Citation** section for details.

---


AAC-MADRL is an **attention-driven, discrete-action, multi-agent actor–critic algorithm** tailored for energy flexibility control at the district level. 

Key characteristics:

- **Paradigm**: follows **Centralized Training with Decentralized Execution (CTDE)**.
- **Actors (πᵢ)**: one per building, with  **discrete probability distribution over actions**.  
  Building devices (domestic hot water storage, electrical storage, heating/cooling systems) are represented as **discretized action classes** (e.g., charging/discharging levels, on/off modulation)
- **Centralized Critic (Q)**: during training, a single critic evaluates joint state–action tuples \(Q(s, a_1, \dots, a_N)\).  
  - It incorporates a **multi-head attention mechanism** to dynamically weight the influence of other agents’ states and actions.  

This makes AAC-MADRL suitable for large residential districts, where **coordination** is crucial.

### Critic Architecture
<img width="1030" height="670" alt="Immagine1" src="https://github.com/user-attachments/assets/600d8194-69ee-483e-8096-9aaf5270a734" />
*The centralized critic computes Q-functions for each agent by embedding each agent’s state–action pair, applying a multi-head attention module to extract the most relevant inter-agent dependencies, and aggregating the attended features before the final Q-value regression layer.*









