# Building Multi-Agent Systems with Google's Agent Development Kit (ADK)

The AI landscape has dramatically shifted toward multi-agent architectures as a fundamental paradigm for building robust, intelligent systems. These architectures transcend traditional single-agent limitations by enabling specialized agents to collaborate, delegate tasks, and solve complex problems through distributed reasoning. Google's **Agent Development Kit (ADK)** represents a significant leap forward in this domain - an industrial-grade, open-source framework meticulously designed for constructing, orchestrating, evaluating, and deploying sophisticated multi-agent systems. While ADK is optimized for Google's ecosystem, its model-agnostic approach ensures flexibility across various LLM providers, allowing developers to create scalable, Python-based agent systems that can be applied to real-world problems.

In this technical deep-dive, I'll dissect the agent typology within ADK, examine the architectural patterns for multi-agent orchestration, and provide a code implementation that demonstrates these concepts in action. I'll also include a detailed sequence chart to illustrate the internal communication flows between agents. Whether you're an AI engineer looking to expand your toolkit or a researcher exploring multi-agent systems, this guide offers both theoretical foundations and practical implementation patterns for leveraging ADK effectively.

---

## 1. Types of ADK Agents

ADK's agent framework provides a well-designed typology that supports various cognitive and orchestration patterns. These agent classes can be composed to create complex systems with specialized capabilities. These agent types are essential for different aspects of system design and provide the building blocks for sophisticated agent architectures.

### **LLM Agents (`LlmAgent`, aliased as `Agent`)**
- **Technical Implementation**: These agents encapsulate an LLM within the ADK framework, providing the cognitive engine for reasoning, planning, and decision-making. Under the hood, ADK formats instructions, context, and queries in a structure optimized for the chosen LLM backend.
- **Architecture Role**: These function as the "thinking" components within your system, handling tasks requiring high cognitive flexibility, contextual understanding, and natural language processing.
- **Internal Structure**: Each LLM agent contains model configuration, system instructions, a description field (useful for agent-to-agent communication), and access control definitions for sub-agents and tools.
- **Use Case**: Critical for user-facing interfaces, complex reasoning tasks, or anywhere requiring high adaptability to unstructured inputs.

### **Workflow Agents**
Workflow agents handle structured processes without needing LLM reasoning for each step, making them deterministic and highly efficient for orchestration. They operate as higher-order control flows:

- **SequentialAgent**: Implements a pipeline architecture where each sub-agent processes the output from its predecessor. This creates a clear transformation chain that's ideal for multi-stage processing pipelines.
  
- **ParallelAgent**: Enables concurrent execution across multiple sub-agents, with the parent agent aggregating results. This pattern significantly reduces latency when independent processing streams can run simultaneously.
  
- **LoopAgent**: Implements iterative refinement patterns by repeatedly executing sub-agents until a specified termination criterion is met. This is particularly valuable for algorithms requiring convergence or multi-step verification.

The efficiency advantage here comes from bypassing LLM invocations for control flow decisions, making these agents ideal for predetermined process workflows where the routing logic is fixed.

### **Custom Agents (`BaseAgent`)**
- **Technical Implementation**: The `BaseAgent` abstract class provides a powerful extension mechanism where you can define custom orchestration logic in Python. This gives you complete control over the execution flow, state management, and agent interaction patterns.
- **Architecture Role**: These agents allow you to implement specialized coordination algorithms, complex decision trees, or integration with external systems that don't fit neatly into the predefined workflow types.
- **Internal Structure**: By implementing the `process()` method, you can create arbitrary control flows, conditional execution strategies, and custom interactions with other agents or tools.
- **Use Case**: Ideal for implementing domain-specific coordination patterns, complex decision trees, or integrating with existing business logic systems that require tight control over the agent execution flow.

The flexibility of mixing these agent types enables construction of sophisticated agent architectures that combine the reasoning capabilities of LLMs with deterministic workflows and custom orchestration logic.

---

## 2. Multi-Agent Structures

The architectural patterns you choose significantly impact system performance, maintainability, and scalability. ADK supports several foundational multi-agent structures, each with distinct characteristics and use cases. Understanding these patterns is crucial for designing effective agent networks.

### **Hierarchical Coordination**
- **Implementation Pattern**: This pattern employs a central coordinator agent (typically an LLM Agent) that routes user queries to specialized sub-agents based on domain expertise or functional capabilities.
- **System Characteristics**: Creates a clear separation of concerns where the coordinator handles task delegation while specialized agents focus on domain-specific reasoning.
- **Performance Considerations**: Adds a coordination layer that increases system latency but significantly enhances specialization capability and maintainability.
- **When to Use**: Ideal for complex applications where clear domain boundaries exist, such as comprehensive customer service systems or multi-domain virtual assistants.
- **Diagram**:
  ```mermaid
  graph TD
      Coordinator --> Sub1
      Coordinator --> Sub2
      Coordinator --> Sub3
  ```

### **Sequential Execution**
- **Implementation Pattern**: Implements a pipeline architecture where each agent's output becomes the input for the next agent in the chain.
- **System Characteristics**: Creates a clear transformation path where each agent adds value or transforms the output from the previous stage.
- **Performance Considerations**: Total latency accumulates with each stage, but enables complex multi-step transformations with clear boundaries between processing stages.
- **When to Use**: Perfect for multi-stage workflows like content generation pipelines (e.g., outline → draft → edit → finalize) or data processing sequences.
- **Diagram**:
  ```mermaid
  graph LR
      Agent1 --> Agent2 --> Agent3
  ```

### **Parallel Execution**
- **Implementation Pattern**: Distributes a task across multiple agents that operate concurrently, with results aggregated afterward.
- **System Characteristics**: Enables simultaneous processing of independent subtasks, maximizing throughput.
- **Performance Considerations**: Overall latency is determined by the slowest agent in the parallel group, but throughput increases linearly with the number of agents.
- **When to Use**: Most effective when tasks can be decomposed into independent subtasks, such as processing multiple documents, analyzing different data streams, or gathering information from various sources simultaneously.
- **Diagram**:
  ```mermaid
  graph TD
      Start --> Agent1
      Start --> Agent2
      Start --> Agent3
  ```

### **Looping Execution**
- **Implementation Pattern**: Implements an iterative refinement pattern where a sequence of agents is repeatedly executed until a termination condition is met.
- **System Characteristics**: Enables continuous improvement or convergence-based tasks through controlled iteration.
- **Performance Considerations**: Total latency depends on the number of iterations required, which may vary based on input complexity.
- **When to Use**: Valuable for tasks requiring progressive refinement, like iterative design processes, optimization problems, or any task requiring multiple passes to reach an acceptable quality level.
- **Diagram**:
  ```mermaid
  graph TD
      Start --> Agent1
      Agent1 --> Agent2
      Agent2 --> Agent3
      Agent3 --> Condition{"Condition Met?"}
      Condition -->|No| Agent1
      Condition -->|Yes| End
  ```

### **Hybrid Architectures**
In production systems, the most effective architectures typically combine these patterns into hybrid structures. For example:
- **Hierarchical-Sequential**: A coordinator routes to different sequential pipelines based on task type
- **Parallel-Loop**: Multiple looping processes execute in parallel to optimize different aspects of a solution
- **Hierarchical-Parallel**: A coordinator delegates to multiple agents that execute in parallel before results are aggregated

Using ADK's `BaseAgent` implementation, you can design custom orchestration patterns that implement these hybrid architectures to suit specific application requirements.

---

## 3. Simple Python Multi-Agent Example

To demonstrate how ADK enables multi-agent collaboration, let's build a simple customer service system. In this example:
- A **coordinator agent** receives user queries.
- Based on the query, it delegates the task to either a **billing agent** (for payment issues) or a **support agent** (for technical problems).
The example code repo: https://github.com/guolisen/docs_code_project/tree/main/adk_multi_agents
the jupyter notebook of https://github.com/guolisen/docs_code_project/tree/main/adk_multi_agents/multi_agent_demo.ipynb can show the example detail.

### **Python Code**
```python
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.genai import types
import google.generativeai as genai
from google.adk.models.lite_llm import LiteLlm
import os
import logging

# Define specialized sub-agents
billing_agent = LlmAgent(
    name="Billing",
    model=LiteLlm(model="gpt-3.5-turbo-0125"),
    instruction="You handle billing and payment-related inquiries.",
    description="Handles billing inquiries."
)

support_agent = LlmAgent(
    name="Support",
    model=LiteLlm(model="gpt-3.5-turbo-0125"),
    instruction="You provide technical support and troubleshooting assistance.",
    description="Handles technical support requests."
)

# Define the coordinator agent
coordinator = LlmAgent(
    name="HelpDeskCoordinator",
    model=LiteLlm(model="gpt-3.5-turbo-0125"),
    instruction="Route user requests: Use Billing agent for payment issues, Support agent for technical problems.",
    description="Main help desk router.",
    sub_agents=[billing_agent, support_agent]
)

# For ADK compatibility, the root agent must be named `root_agent`
root_agent = coordinator

# Set up the runner
runner = Runner(
        app_name="test_agent",
        agent=root_agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService())
```

### **Support Agent Query Example**

```python
# Test with a technical support query
support_query = "I can't access my email account, it says my password is incorrect, help to give advice"
print(f"Query: {support_query}")

# Run the query and get response and events
support_response, support_events = test_query(support_query)
print(f"Response: {support_response}")

# Extract the agent invoke path
support_path = extract_agent_path(support_events)
print(f"Agent Path: {support_path}")
```

Sample output:
```
Query: I can't access my email account, it says my password is incorrect, help to give advice
Agent Path: ['HelpDeskCoordinator', 'Support']
```

### **Billing Agent Query Example**

```python
# Test with a billing query
billing_query = "I was charged twice for my subscription last month"
print(f"Query: {billing_query}")

# Run the query and get response and events
billing_response, billing_events = test_query(billing_query)
print(f"Response: {billing_response}")

# Extract the agent invoke path
billing_path = extract_agent_path(billing_events)
print(f"Agent Path: {billing_path}")
```

Sample output:
```
Query: I was charged twice for my subscription last month
Agent Path: ['HelpDeskCoordinator', 'Billing']
```

This demonstrates how the coordinator agent intelligently routes different types of queries to the appropriate specialized agent.

---

## 4. Explanation with Sequence Chart

Let's break down how the multi-agent system processes the user query "I can't log in" using a sequence chart.

### **Sequence of Events**
1. **User sends a message** to the coordinator agent: "I can't log in."
2. **Coordinator's LLM processes the message** and determines that it is a technical support request.
3. **Coordinator delegates the task** by generating a function call to `transfer_to_agent` with `agent_name='Support'`.
4. **The framework routes execution** to the support agent.
5. **Support agent processes the query** and generates a response, such as "Please try resetting your password or check your internet connection."
6. **The response is yielded back** to the runner and sent to the user.

### **Sequence Diagram**
```mermaid
sequenceDiagram
    participant User
    participant Coordinator
    participant LLM
    participant SupportAgent
    User->>Coordinator: "I can't log in"
    Coordinator->>LLM: Process query
    LLM->>Coordinator: Transfer to 'Support'
    Coordinator->>SupportAgent: Delegate task
    SupportAgent->>LLM: Generate response
    LLM->>SupportAgent: "Please try resetting your password..."
    SupportAgent->>Coordinator: Response
    Coordinator->>User: "Please try resetting your password..."
```

This sequence demonstrates how ADK's hierarchical coordination enables intelligent task delegation, allowing specialized agents to handle specific types of queries efficiently.

---

## Conclusion

Google's Agent Development Kit (ADK) is a game-changer for developers looking to build scalable, multi-agent AI systems. With its support for LLM-powered agents, structured workflow agents, and custom orchestration logic, ADK provides the flexibility needed to tackle a wide range of applications—from customer service bots to complex data pipelines. By understanding the different agent types and multi-agent structures, you can design systems that are both powerful and easy to manage.

The example provided illustrates just one of many possibilities. As you explore ADK further, you'll discover even more ways to combine agents, tools, and workflows to solve real-world problems. So, dive into the [ADK documentation](https://google.github.io/adk-docs/) and start building your own multi-agent applications today!

---

**Sources**  
This article is based on information from the official ADK documentation and related resources. For more details, refer to the [ADK Multi-Agent Systems page](https://google.github.io/adk-docs/agents/multi-agents/).

---

### Notes on Mermaid Diagrams
- **Rendering**: The Mermaid code blocks can be rendered into diagrams using tools like the Mermaid Live Editor or the Mermaid CLI.
- **Platform Compatibility**: If publishing on platforms that do not support Mermaid directly (e.g., Medium), you can generate images from the Mermaid code and embed them in the article.
