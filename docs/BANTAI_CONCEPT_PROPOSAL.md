# BANTAI Concept Proposal

**Institution:** Technological Institute of the Philippines — Quezon City  
**Department:** College of Computer Studies, Computer Science Department  
**Course:** CS 403 – Intelligent Agents, First Semester S.Y. 2026–2027  
**Faculty In-Charge:** Mr. Jess N. Garcia  

---

## (1) Proponents and Roles

| Name | Specified Role |
| :--- | :--- |
| **Aguila, Sophia Mae** | Integration & Quality Assurance / Integration Developer |
| **Gumabon, Joshua Philip** | Front-End / UI/UX Developer |
| **Lalamoro, Justine Mark** | Model Trainer / Developer |
| **Santiago, Samuel Rei** | Project Manager / Scripting & Analytics Developer |
| **Soriano, Jhun Deniel** | Full-Stack / Lead Web Developer |

---

## (2) Project Title
**BANTAI: An Intelligent Agent for Street Issue Detection and Hotspot Prioritization**

---

## (3) Background of the Study

Rapid urbanization, population growth, and expanding economic activity bring significant everyday challenges for local government units, particularly in keeping neighborhoods clean and maintaining safe road conditions (Reddy et al., 2025). In busy municipal areas, staying on top of waste management and identifying road degradation requires substantial workforce resources and continuous manual oversight. Even with local environmental ordinances and community sanitation guidelines in place, local personnel frequently run into practical bottlenecks when trying to identify, track, and address localized issues in a timely manner (Gamao & Caelian, 2023).

For many communities, the current monitoring framework relies heavily on manual field patrols, reactive resident complaints, and routine spot checks. To begin with, relying solely on manual inspection makes covering wide road networks difficult, as staff must physically survey areas to spot roadside garbage piles, potholes, and surface damage—a process that is time-consuming, labor-intensive, and prone to human oversight (Maske, 2025). Additionally, incident reports often become fragmented and repetitive, as residents and field staff frequently report the exact same issue through separate channels, which quickly clutters local databases with duplicate entries when no central mechanism exists to merge them.

Beyond reporting inefficiencies, local officials face ongoing challenges when allocating maintenance teams and equipment, as they lack an objective, data-driven way to prioritize issues based on how often or where they are being reported. Furthermore, existing operations rarely connect modern automated detection tools with a central web interface, leaving leaders without clear visual summaries or neighborhood trend data to help plan proactive cleanups and road repairs (Reddy et al., 2025). To help address these practical limitations, this study proposes an intelligent agent-based system designed to automatically detect anomalies through pretrained computer vision models, combine duplicate reports into consolidated records, and dynamically highlight high-priority hotspot locations based on report frequency across target communities.

---

## (4) Project Objectives

The researchers aim to develop an intelligent agent-based system that identifies selected environmental and road infrastructure anomalies and prioritizes reported incidents and hotspot locations within a barangay. Through the creation of a prototype, the system aims to support barangay personnel and officials in planning cleanup, inspection, and maintenance activities through computer vision-assisted detection and frequency-based decision support.

Specifically, the study aims:
1. To implement pretrained computer vision models capable of identifying selected environmental and road infrastructure anomalies, particularly garbage piles, potholes, and road surface damage, through user-submitted photographs and prerecorded video footage simulating CCTV-based anomaly monitoring.
2. To develop a web-based incident reporting and management system that records identified anomalies, their locations, timestamps, supporting images, detection results, and incident statuses, while providing a mechanism for consolidating multiple observations of the same incident to minimize duplicate records.
3. To design a frequency-based intelligent agent that automatically aggregates validated incident reports and observation records to dynamically prioritize individual incidents and identify hotspot locations based on reporting frequency, using report recency as a secondary criterion when rankings are equal.
4. To develop an accessible and user-friendly web dashboard that displays identified incidents, prioritized hotspot locations, incident statuses, and descriptive trends based on anomaly categories, locations, and reporting frequency to support barangay cleanup, inspection, and maintenance decisions.
5. To evaluate the developed system in terms of computer vision detection performance for each supported anomaly category, incident consolidation accuracy, hotspot prioritization consistency, processing time, usability, and basic accessibility.

---

## (5) Expected Output

The project is expected to produce a functional prototype of an intelligent agent-based environmental and road infrastructure anomaly detection and decision-support system. The system will integrate pretrained computer vision models, a web-based incident reporting and management platform, and a frequency-based intelligent agent to identify, organize, and prioritize reported anomalies. The computer vision component will utilize pretrained object detection models, such as RF-DETR Nano and Grounding DINO Tiny, to identify selected anomalies, particularly potholes, road surface damage, and roadside garbage accumulation. Anomaly identification will be demonstrated through two methods: user-submitted photographs and prerecorded video footage simulating CCTV-based monitoring, without requiring continuous live camera integration.

The system will provide a centralized database that stores incident information, including identified anomaly categories, reported locations, timestamps, supporting images, detection results, and incident statuses. Submitted reports and video-based observations will undergo validation and incident consolidation to minimize duplicate records. A frequency-based intelligent agent will automatically evaluate validated incident records and generate dynamic priority rankings based primarily on the number of valid supporting reports, using report recency as a secondary criterion for resolving equal rankings. The agent will also aggregate reporting frequency according to predefined streets or monitoring areas to identify and rank reported-anomaly hotspots. These rankings will be updated as new validated reports are received or existing incident statuses change.

An accessible and user-friendly web dashboard will serve as the primary interface for barangay personnel and officials. It will display identified anomalies, incident details, priority rankings, hotspot locations, and incident statuses, alongside descriptive summaries and visualizations of anomaly categories, reporting frequency, and location-based trends. Through these features, the system is expected to provide organized and data-driven information that supports the planning of cleanup, inspection, and maintenance activities. Priority rankings will represent the frequency of validated reports rather than the physical severity or actual danger of an incident, allowing barangay personnel to retain responsibility for determining appropriate actions.

The completed prototype will undergo evaluation to determine its detection performance across supported anomaly categories, effectiveness of incident consolidation, consistency of frequency-based prioritization and hotspot identification, and processing time. The web interface will also be assessed for usability and basic accessibility features, including readable content, sufficient visual contrast, properly labeled controls, and keyboard navigation. The resulting system will demonstrate the feasibility of combining pretrained computer vision models with a frequency-based intelligent agent to provide practical decision support for localized environmental and road infrastructure monitoring.

---

## References

- Gamao, R. A., & Caelian, M. V. (2023). Implementation and challenges of solid waste management in communities of a component city in the Philippines. *Technium Social Sciences Journal*, 44, 28–40. https://doi.org/10.47577/tssj.v44i1.8897
- Maske, Y. (2025). Urban civic issues image dataset: Potholes and garbage (QR4Change) (Version 2) [Data set]. *Mendeley Data*. https://doi.org/10.17632/zndzygc3p3.2
- Reddy, P. S., Battula, B., Pasha, M. F., Reddy, G. B. S., Bimbasri, P., & Virinchi, E. (2025). YOLO-based system for efficient garbage and pothole detection and reporting. *2025 Third International Conference on Networks, Multimedia and Information Technology (NMITCON)*, IEEE. https://doi.org/10.1109/NMITCON65824.2025.11188066
