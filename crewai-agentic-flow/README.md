
# Event Planning Multi-Agent System

This project implements a multi-agent system using **CrewAI** and **Ollama** models to automate the event planning process. It defines three agents:

1. **Event Planning Agent**: Responsible for creating event plans, schedules, and itineraries.
2. **Vendor Coordination Agent**: Manages communication and coordination with vendors.
3. **Budget Management Agent**: Ensures the event stays within budget while maintaining quality.

## Installation

To set up the project, install the necessary dependencies:

```bash
pip3 install crewai==0.105.0 crewai_tools ollama
```

## Usage

Once the dependencies are installed, you can run the project and interact with the event planning system via the FastAPI app.

### Access the API

After running the app, you can access the event planning system by visiting [http://localhost:8080/chain/playground/](http://localhost:8080/chain/playground/) in your browser.

- You can enter a topic (e.g., "wedding", "conference") and receive relevant event planning responses or suggestions based on the topic.

## Features

- **Event Planning**: The system can create detailed event plans and schedules based on the given topic, including allocating resources and identifying potential risks.
- **Vendor Coordination**: The vendor coordination agent helps to manage communication with vendors (such as catering, decorations, audio-visual equipment) and ensures timely deliveries.
- **Budget Management**: The budget management agent ensures that the event stays within the allocated budget while still meeting quality expectations.

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **CrewAI** for enabling multi-agent orchestration.
- **Ollama** for providing the large language model for event planning.




