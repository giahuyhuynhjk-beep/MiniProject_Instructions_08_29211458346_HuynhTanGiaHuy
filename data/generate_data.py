"""Dataset generator and validator for Tech Blog and News Article Recommender.
CS467 Advanced AI - Assignment 15 (MP-A08)
Student: Huynh Tan Gia Huy (ID 29211458346)
"""

import random
import pandas as pd
import numpy as np
from pathlib import Path

# Seed for deterministic generation
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent

ITEMS_DATA = [
    # 1. Cybersecurity (A001 - A005)
    {
        "item_id": "A001",
        "headline": "Zero-Day Privilege Escalation Vulnerability Discovered in Linux Kernel Netfilter",
        "topic": "Cybersecurity",
        "publish_date": "2026-02-14",
        "summary_text": "Security researchers uncovered a heap out-of-bounds write zero-day flaw in Linux kernel Netfilter subsystems. The vulnerability allows unprivileged local attackers to bypass kernel security modules and escalate privileges to root. Distribution maintainers released an urgent security advisory and emergency kernel patch to remediate active exploitation.",
        "tags": "linux;kernel;vulnerability;zero-day;privilege-escalation;netfilter;patch;security"
    },
    {
        "item_id": "A002",
        "headline": "Black Basta Ransomware Syndicate Deploys Novel Evasion Tactics Against Enterprise Networks",
        "topic": "Cybersecurity",
        "publish_date": "2026-03-02",
        "summary_text": "Threat intelligence analysts detected updated variants of Black Basta ransomware leveraging living-off-the-land binaries to circumvent endpoint detection. The malware rapidly encrypts virtualized storage and exfiltrates corporate credentials before triggering extortion demands. Cybersecurity agencies recommend strict network segmentation and rapid credential rotation.",
        "tags": "ransomware;cybersecurity;malware;threat-intelligence;edr;encryption;hardening;mitigation"
    },
    {
        "item_id": "A003",
        "headline": "Memory-Safe Hardening Frameworks Mandated for Critical Infrastructure Defense",
        "topic": "Cybersecurity",
        "publish_date": "2026-01-28",
        "summary_text": "Federal cybersecurity regulators published new guidelines mandating memory-safe architectures across defense and banking infrastructures. Systems engineers are replacing legacy C/C++ services with hardened Rust microservices to eliminate buffer overflows and dangling pointer zero-days. Audits show a dramatic reduction in exploitable memory safety vulnerabilities.",
        "tags": "security;hardening;memory-safety;infrastructure;zero-day;defense;compliance;rust"
    },
    {
        "item_id": "A004",
        "headline": "State-Sponsored APT Groups Exploit Unpatched Edge Gateway Zero-Days Globally",
        "topic": "Cybersecurity",
        "publish_date": "2026-04-10",
        "summary_text": "Sophisticated threat actors compromised perimeter network firewalls by targeting undocumented authentication bypass zero-day flaws. Intruders deployed stealthy backdoors directly into device firmware, surviving standard system reboots. Security teams are advised to isolate management interfaces and audit external edge appliance firmware.",
        "tags": "cybersecurity;zero-day;apt;firewall;firmware;backdoor;exploit;network-defense"
    },
    {
        "item_id": "A005",
        "headline": "Automated Vulnerability Auditing Identifies Critical Flaws in Open-Source Cryptography Libraries",
        "topic": "Cybersecurity",
        "publish_date": "2026-03-22",
        "summary_text": "A comprehensive automated audit utilizing symbolic execution identified timing side-channel leaks within widely used cryptographic packages. The vulnerabilities could allow adversaries to recover secret decryption keys under specific network conditions. Upstream open-source maintainers coordinated patched releases across major enterprise Linux distributions.",
        "tags": "cryptography;vulnerability;audit;security;open-source;side-channel;patch;cve"
    },

    # 2. Linux (A006 - A010)
    {
        "item_id": "A006",
        "headline": "Linux 6.14 Kernel Release Brings Real-Time Preemption and Enhanced eBPF Observability",
        "topic": "Linux",
        "publish_date": "2026-02-18",
        "summary_text": "Linus Torvalds announced the general availability of the Linux 6.14 kernel featuring fully merged PREEMPT_RT real-time locking primitives. The update expands eBPF tracing capabilities for memory subsystems and introduces security hardening patches against speculative execution bugs. Systems programmers reported lower scheduler latency under intense server workloads.",
        "tags": "linux;kernel;ebpf;real-time;observability;scheduler;hardening;sysadmin"
    },
    {
        "item_id": "A007",
        "headline": "Arch Linux and Fedora Adopt Unified Kernel Images for Secure Boot Hardening",
        "topic": "Linux",
        "publish_date": "2026-01-19",
        "summary_text": "Major Linux distributions are transitioning towards Unified Kernel Images combining the kernel, initramfs, and command line into a single signed UEFI binary. This architecture bolsters tamper-resistance and eliminates vulnerabilities in multi-stage bootloaders. The initiative modernizes Linux desktop security and simplifies system recovery mechanisms.",
        "tags": "linux;distro;secure-boot;uefi;fedora;arch;hardening;kernel"
    },
    {
        "item_id": "A008",
        "headline": "Modern Linux Server Hardening: Best Practices for Seccomp, AppArmor, and Namespaces",
        "topic": "Linux",
        "publish_date": "2026-03-15",
        "summary_text": "Hardening production Linux servers requires defense-in-depth through container sandboxing, restricted syscall filters, and mandatory access control policies. System administrators demonstrate configuring granular AppArmor profiles and seccomp system call filtering to neutralize zero-day privilege escalations. Comprehensive audit logs further ensure rapid detection of anomalous shell behavior.",
        "tags": "linux;hardening;seccomp;apparmor;namespaces;security;sysadmin;privilege-escalation"
    },
    {
        "item_id": "A009",
        "headline": "Next-Generation Storage Architectures in Linux: Evaluating Bcachefs Stability and Performance",
        "topic": "Linux",
        "publish_date": "2026-04-05",
        "summary_text": "The Linux kernel storage ecosystem advances with extensive multi-device tiering and checksumming benchmarks for the Bcachefs filesystem. Engineers compare Bcachefs against Btrfs and ZFS under heavy database input-output workloads. Results indicate robust data corruption recovery and low overhead during background snapshot maintenance.",
        "tags": "linux;kernel;filesystem;storage;bcachefs;performance;btrfs;sysadmin"
    },
    {
        "item_id": "A010",
        "headline": "Debian Releases Landmark Security Overhaul Mitigating Hardware-Level Side-Channel Attacks",
        "topic": "Linux",
        "publish_date": "2026-02-27",
        "summary_text": "Debian security maintainers published a point release delivering comprehensive kernel hardening against transient execution microarchitectural flaws. The release incorporates backported kernel patches that isolate virtual memory addresses between unprivileged user processes. Server administrators are urged to apply the updates to safeguard multi-tenant hosting nodes.",
        "tags": "linux;debian;distro;security;patch;kernel;hardening;side-channel"
    },

    # 3. Open-Source LLMs (A011 - A015)
    {
        "item_id": "A011",
        "headline": "Running Quantized Open-Source LLMs on Consumer Laptops with llama.cpp and Ollama",
        "topic": "Open-Source LLMs",
        "publish_date": "2026-01-22",
        "summary_text": "Developers are executing high-performance local AI inference on lightweight laptops using llama.cpp and Ollama runtimes. By utilizing 4-bit and 5-bit GGUF quantization, multi-billion parameter open-weight models fit comfortably inside modest consumer CPU and GPU RAM. This local workflow delivers total data privacy and zero cloud API dependency.",
        "tags": "llm;open-source;quantization;llamacpp;ollama;laptop;local-inference;cpu;gpu;gguf"
    },
    {
        "item_id": "A012",
        "headline": "Small Open-Weight Language Models Achieve Parity with Frontier Systems on Consumer Hardware",
        "topic": "Open-Source LLMs",
        "publish_date": "2026-02-11",
        "summary_text": "Recent 3B to 8B open-weight language models demonstrate remarkable code generation and reasoning capabilities that rival closed cloud systems. Through aggressive synthetic data fine-tuning and architectural optimizations, these compact models execute with lightning speed on local laptop hardware. Independent benchmarks highlight exceptional performance without requiring enterprise server racks.",
        "tags": "llm;open-source;open-weight;small-models;consumer-hardware;local-ai;laptop;benchmarks"
    },
    {
        "item_id": "A013",
        "headline": "4-Bit GGUF Quantization Techniques for Efficient Local LLM Inference on CPU and Laptop GPU",
        "topic": "Open-Source LLMs",
        "publish_date": "2026-03-08",
        "summary_text": "A technical breakdown explores state-of-the-art GGUF and AWQ quantization algorithms that compress open-source LLM weights with minimal perplexity degradation. The approach enables smooth text generation on integrated laptop GPUs and commodity multi-core CPUs. Memory bandwidth bottlenecks are mitigated using SIMD vector instructions and cached KV tensors.",
        "tags": "quantization;gguf;llm;open-source;local-inference;laptop;cpu;gpu;memory-optimization"
    },
    {
        "item_id": "A014",
        "headline": "Privacy-Preserving AI: Deploying Open-Source Local Language Models on Apple Silicon and Laptops",
        "topic": "Open-Source LLMs",
        "publish_date": "2026-03-29",
        "summary_text": "Software teams are adopting offline open-source language models to index sensitive corporate codebases without exposing secrets to external APIs. Unified memory architectures on modern laptops and MacBooks enable zero-copy tensor execution for 70B quantized models at interactive token speeds. The local pipeline ensures complete GDPR and intellectual property compliance.",
        "tags": "llm;open-source;privacy;apple-silicon;laptop;local-inference;unified-memory;offline-ai"
    },
    {
        "item_id": "A015",
        "headline": "Low-Latency Speculative Decoding Enables High-Throughput Open-Source LLM Execution Locally",
        "topic": "Open-Source LLMs",
        "publish_date": "2026-04-18",
        "summary_text": "Speculative decoding has emerged as a crucial accelerator for local open-source LLM inference on consumer workstations. By pairing a tiny draft model with a larger target model, inference engines generate multiple tokens per memory read cycle. Open-source implementations like llama.cpp and vLLM achieve a threefold speedup on ordinary laptop processors.",
        "tags": "llm;speculative-decoding;open-source;latency;local-inference;throughput;laptop;vllm"
    },

    # 4. AI/ML (A016 - A020)
    {
        "item_id": "A016",
        "headline": "Direct Preference Optimization Replaces RLHF for Post-Training Frontier Foundation Models",
        "topic": "AI/ML",
        "publish_date": "2026-02-04",
        "summary_text": "Machine learning practitioners are widely adopting Direct Preference Optimization (DPO) and its variants to align foundation models without unstable reinforcement learning loops. The closed-form objective directly optimizes implicit reward representations on paired human preference data. Experiments show increased conversational coherence and decreased reward hacking during model fine-tuning.",
        "tags": "ai;ml;alignment;dpo;rlhf;foundation-models;fine-tuning;deep-learning"
    },
    {
        "item_id": "A017",
        "headline": "Multi-Modal Vision Transformers Master Fine-Grained Document Parsing and Scene Understanding",
        "topic": "AI/ML",
        "publish_date": "2026-03-12",
        "summary_text": "New multi-modal vision-language architectures integrate cross-attention mechanisms to seamlessly parse complex technical diagrams, financial tables, and natural images. The models demonstrate zero-shot visual question answering by bridging visual token representations with dense text embeddings. Industrial research labs report substantial accuracy leaps in automated document workflows.",
        "tags": "ai;ml;computer-vision;transformers;multimodal;ocr;deep-learning;document-ai"
    },
    {
        "item_id": "A018",
        "headline": "Latent Diffusion Models Revolutionize High-Resolution Synthetic Data Generation for Computer Vision",
        "topic": "AI/ML",
        "publish_date": "2026-01-30",
        "summary_text": "Researchers harness text-conditioned latent diffusion models to generate photorealistic synthetic training datasets for autonomous robotics and medical imaging. Controlled generation pipelines mitigate class imbalance and eliminate edge-case data deficits in production vision models. Rigorous validation confirms that synthetic pre-training rivals costly manual labeling campaigns.",
        "tags": "ai;ml;diffusion-models;generative-ai;synthetic-data;computer-vision;robotics;training"
    },
    {
        "item_id": "A019",
        "headline": "Approximate Nearest Neighbor Search Algorithms Scale Vector Databases to Billions of Embeddings",
        "topic": "AI/ML",
        "publish_date": "2026-04-02",
        "summary_text": "Vector database engineers implemented hierarchical navigable small-world (HNSW) and product quantization indexing to scale semantic search across billion-scale embedding collections. The optimized indexing strategies balance query recall with sub-millisecond retrieval latency for retrieval-augmented generation pipelines. Memory-efficient graph pruning techniques drastically reduce infrastructure hosting costs.",
        "tags": "ai;ml;vector-database;embeddings;hnsw;rag;semantic-search;ann;retrieval"
    },
    {
        "item_id": "A020",
        "headline": "Sparse Mixture-of-Experts Architectures Slash Inference Costs in Large-Scale AI Systems",
        "topic": "AI/ML",
        "publish_date": "2026-02-25",
        "summary_text": "Deep learning researchers are transitioning monolithic neural networks toward sparse mixture-of-experts (MoE) topologies with dynamically routed sub-networks. By activating only a fraction of total model parameters per token, MoE models achieve state-of-the-art capability at a fraction of standard FLOPs. Serving architectures leverage expert parallelism to optimize distributed GPU cluster utilization.",
        "tags": "ai;ml;mixture-of-experts;moe;deep-learning;inference-efficiency;gpu;neural-networks"
    },

    # 5. Cloud/DevOps (A021 - A025)
    {
        "item_id": "A021",
        "headline": "Kubernetes 1.33 Introduces Declarative Node Auto-Scaling and Enhanced Resource Slicing",
        "topic": "Cloud/DevOps",
        "publish_date": "2026-03-18",
        "summary_text": "The Cloud Native Computing Foundation released Kubernetes 1.33 with native declarative cluster auto-scaling and dynamic GPU resource slicing. Platform engineers can now partition hardware accelerators across multi-tenant workloads with sub-device precision. The release also hardens internal API server admission controllers against privilege escalation vectors.",
        "tags": "cloud;devops;kubernetes;containers;autoscaling;infrastructure;gpu-slicing;cncf"
    },
    {
        "item_id": "A022",
        "headline": "GitOps Best Practices: Managing Multi-Cluster Environments with ArgoCD and Terraform",
        "topic": "Cloud/DevOps",
        "publish_date": "2026-01-16",
        "summary_text": "Enterprise DevOps teams are consolidating heterogeneous cloud infrastructure using GitOps declarative state repositories managed by ArgoCD and OpenTofu/Terraform. Automated drift detection continuously synchronizes configuration changes across global production Kubernetes clusters. The approach establishes immutable audit logs and facilitates rapid disaster recovery rollback.",
        "tags": "cloud;devops;gitops;argocd;terraform;infrastructure-as-code;kubernetes;automation"
    },
    {
        "item_id": "A023",
        "headline": "OpenTelemetry Becomes the Gold Standard for End-to-End Distributed Tracing in Microservices",
        "topic": "Cloud/DevOps",
        "publish_date": "2026-02-22",
        "summary_text": "Site reliability engineers have unified metrics, logs, and distributed traces under vendor-neutral OpenTelemetry standards. Zero-code auto-instrumentation agents capture asynchronous microservice RPC spans without degrading application request throughput. Real-time telemetry dashboards enable rapid isolation of performance bottlenecks across complex cloud topologies.",
        "tags": "cloud;devops;opentelemetry;observability;tracing;microservices;monitoring;sre"
    },
    {
        "item_id": "A024",
        "headline": "Zero-Trust Service Meshes: Implementing Mutual TLS and Granular IAM Across Hybrid Clouds",
        "topic": "Cloud/DevOps",
        "publish_date": "2026-03-30",
        "summary_text": "Securing distributed cloud architectures requires ambient service mesh infrastructure that enforces mutual TLS encryption and identity-aware routing. Security architects integrate SPIFFE/SPIRE cryptographic identities to eliminate long-lived service account tokens across public cloud VPCs. Granular Layer-7 authorization policies prevent lateral movement by malicious actors.",
        "tags": "cloud;devops;service-mesh;zero-trust;mtls;security;iam;spiffe;microservices"
    },
    {
        "item_id": "A025",
        "headline": "Serverless Container Architectures Minimize Cold Starts and Optimize Cloud Compute Budgets",
        "topic": "Cloud/DevOps",
        "publish_date": "2026-04-12",
        "summary_text": "Modern serverless container platforms deploy microVM snapshot restoration to reduce cold-start execution delays down to single-digit milliseconds. Financial operations teams leverage scale-to-zero billing models to drastically reduce idle compute spending across development and staging environments. Benchmarks demonstrate excellent elastic scaling during sudden traffic spikes.",
        "tags": "cloud;devops;serverless;containers;microvm;finops;cloud-native;scalability"
    },

    # 6. Web Development (A026 - A030)
    {
        "item_id": "A026",
        "headline": "React Server Components and Next.js Architecture: Maximizing Core Web Vitals",
        "topic": "Web Development",
        "publish_date": "2026-02-08",
        "summary_text": "Frontend engineers leverage React Server Components and modern streaming SSR in Next.js to drastically shrink client-side JavaScript bundle sizes. Rendering data-intensive UI components directly on edge servers yields near-instantaneous Largest Contentful Paint metrics. The design pattern simplifies backend data fetching while eliminating hydration mismatches.",
        "tags": "webdev;react;nextjs;frontend;javascript;web-performance;ssr;html"
    },
    {
        "item_id": "A027",
        "headline": "Building High-Performance Web Applications with WebAssembly and Rust in the Browser",
        "topic": "Web Development",
        "publish_date": "2026-03-14",
        "summary_text": "Complex in-browser applications such as photo editors, audio synthesizers, and CAD modeling suites are adopting Rust compiled to WebAssembly. The Wasm threads and SIMD proposals unlock near-native CPU throughput within standard web browser sandboxes. Developers maintain predictable frame rates while processing multi-gigabyte client-side assets.",
        "tags": "webdev;webassembly;wasm;rust;performance;browser;frontend;javascript"
    },
    {
        "item_id": "A028",
        "headline": "Modern CSS Architectures: Container Queries, Cascade Layers, and Subgrid in Production",
        "topic": "Web Development",
        "publish_date": "2026-01-25",
        "summary_text": "The evolution of native CSS specifications has reduced the industry's reliance on complex utility CSS frameworks and preprocessors. Frontend designers employ container queries to construct truly modular responsive UI components that adapt based on parent element dimensions. Cascade layers provide declarative control over style specificity in large enterprise web applications.",
        "tags": "webdev;css;frontend;responsive-design;ui-ux;web-standards;styling;html"
    },
    {
        "item_id": "A029",
        "headline": "TypeScript 5.8 Deep Dive: Advanced Type-Level Metaprogramming and Compilation Optimizations",
        "topic": "Web Development",
        "publish_date": "2026-04-04",
        "summary_text": "The TypeScript engineering team shipped version 5.8 with faster type-checking algorithms and enhanced support for const type parameters. Fullstack developers utilize recursive conditional types to construct compile-time safe database queries and API routing contracts. The compiler optimizations significantly decrease build times across sprawling enterprise monorepos.",
        "tags": "webdev;typescript;javascript;frontend;type-safety;metaprogramming;tooling"
    },
    {
        "item_id": "A030",
        "headline": "Transitioning from REST to Federated GraphQL and gRPC-Web for Low-Latency APIs",
        "topic": "Web Development",
        "publish_date": "2026-03-26",
        "summary_text": "Software architects are modernizing legacy REST interfaces by deploying federated GraphQL supergraphs paired with compact gRPC-Web binary transports. Client applications query precisely the relational attributes they require in a single round-trip, ending over-fetching. Strong schema contracts foster seamless cross-team development across web and mobile surfaces.",
        "tags": "webdev;graphql;grpc;api;rest;backend;microservices;web-architecture"
    },

    # 7. Mobile (A031 - A035)
    {
        "item_id": "A031",
        "headline": "Android 16 Introduces Enhanced Sandboxing, Predictive Back Navigation, and Jetpack Compose 2.0",
        "topic": "Mobile",
        "publish_date": "2026-02-16",
        "summary_text": "Google previewed Android 16 with stricter privacy sandboxes that isolate advertising identifiers and background app telemetry. The operating system refines predictive back gestures and integrates seamlessly with Jetpack Compose 2.0 declarative UI toolkits. Android developers benefit from enhanced compile-time layout inspection and smoother animations.",
        "tags": "mobile;android;jetpack-compose;kotlin;privacy;ui-ux;google;app-development"
    },
    {
        "item_id": "A032",
        "headline": "iOS 19 Architecture: Optimizing Swift Concurrency and Declarative SwiftUI Rendering",
        "topic": "Mobile",
        "publish_date": "2026-03-20",
        "summary_text": "Apple released developer toolchains for iOS 19 featuring compile-time data race safety enforced across all Swift structured concurrency actors. The SwiftUI rendering pipeline incorporates hardware-accelerated metal shaders for fluid interactive transitions on ProMotion displays. Engineers praise improved memory isolation that virtually eliminates concurrency crashes in production.",
        "tags": "mobile;ios;swift;swiftui;apple;concurrency;ui-ux;mobile-architecture"
    },
    {
        "item_id": "A033",
        "headline": "Cross-Platform Performance Benchmarks: Flutter Impeller Engine vs React Native New Architecture",
        "topic": "Mobile",
        "publish_date": "2026-01-20",
        "summary_text": "A comprehensive benchmarking study evaluates mobile UI performance across cross-platform frameworks on high-refresh-rate smartphones. Flutter's Impeller graphics engine eliminates runtime shader compilation jank, while React Native's Fabric renderer ensures synchronous JavaScript-to-native UI updates. Both ecosystems demonstrate near-native benchmark parity for data-heavy consumer applications.",
        "tags": "mobile;cross-platform;flutter;react-native;performance;benchmarks;graphics;ui"
    },
    {
        "item_id": "A034",
        "headline": "Hardening Mobile Applications Against Reverse Engineering and Runtime Memory Tampering",
        "topic": "Mobile",
        "publish_date": "2026-04-08",
        "summary_text": "Mobile application security engineers implement advanced anti-tampering defenses including jailbreak/root detection, runtime integrity checks, and obfuscation. Dynamic instrumentation tools like Frida are detected and neutralized before sensitive cryptographic keys can be extracted from app memory. Financial institutions mandate rigorous compliance with OWASP MASVS benchmarks.",
        "tags": "mobile;security;hardening;reverse-engineering;anti-tampering;android;ios;app-security"
    },
    {
        "item_id": "A035",
        "headline": "Battery-Conscious Mobile Engineering: Background Task Scheduling and Display Refresh Tuning",
        "topic": "Mobile",
        "publish_date": "2026-03-05",
        "summary_text": "Conserving mobile battery life requires disciplined scheduling of background synchronization tasks and intelligent throttling of location updates. Mobile engineers adopt opportunistic network batching and adaptive frame-rate governors that drop refresh frequencies during static reading sessions. Device telemetry shows up to a forty percent reduction in energy consumption.",
        "tags": "mobile;battery-optimization;energy-efficiency;android;ios;performance;optimization"
    },

    # 8. Hardware/Chips (A036 - A040)
    {
        "item_id": "A036",
        "headline": "RISC-V Custom Silicon Challenges ARM Dominance in High-Performance Edge Computing",
        "topic": "Hardware/Chips",
        "publish_date": "2026-02-12",
        "summary_text": "Semiconductor startups are launching modular RISC-V processor designs featuring custom vector extensions tailored for embedded artificial intelligence. The royalty-free open instruction set architecture allows silicon engineers to tailor microarchitectures for specific edge workloads. Industry analysts project substantial market share gains across automotive and industrial sectors.",
        "tags": "hardware;chips;riscv;arm;semiconductor;edge-computing;processors;architecture"
    },
    {
        "item_id": "A037",
        "headline": "Next-Generation NPU Silicon Accelerates Matrix Multiplication for On-Device Neural Processing",
        "topic": "Hardware/Chips",
        "publish_date": "2026-03-24",
        "summary_text": "Chipmakers unveiled dedicated Neural Processing Units (NPUs) delivering hundreds of TOPS of int4 and int8 matrix multiplication power on consumer laptops. These specialized silicon blocks handle concurrent voice synthesis, video segmentation, and local LLM token generation without draining the host battery. Tightly coupled SRAM architectures maximize memory bandwidth per watt.",
        "tags": "hardware;chips;npu;neural-processing;matrix-multiplication;laptop;edge-ai;semiconductor"
    },
    {
        "item_id": "A038",
        "headline": "Advanced 2nm Semiconductor Fabrication Overcomes Quantum Tunneling Challenges",
        "topic": "Hardware/Chips",
        "publish_date": "2026-01-24",
        "summary_text": "Leading semiconductor foundries announced commercial mass production readiness for 2nm process nodes utilizing Gate-All-Around nanosheet transistors. The architectural transition mitigates parasitic leakage currents and enhances drive strength compared to legacy FinFET transistors. The breakthrough paves the way for higher clock frequencies and improved energy efficiency in next-generation processors.",
        "tags": "hardware;chips;semiconductor;nanotechnology;fabrication;transistors;foundry"
    },
    {
        "item_id": "A039",
        "headline": "GPU Tensor Core Architectural Innovations Drive Exponential Throughput in Matrix Arithmetic",
        "topic": "Hardware/Chips",
        "publish_date": "2026-04-14",
        "summary_text": "The newest enterprise GPU architectures integrate asynchronous tensor cores with native FP4 precision formats to accelerate deep learning training clusters. High-bandwidth HBM4 memory stacks deliver terabytes per second of memory throughput directly to computing pipelines. The hardware enhancements reduce cluster training time for trillion-parameter foundation models.",
        "tags": "hardware;chips;gpu;tensor-cores;semiconductor;matrix-arithmetic;hbm;deep-learning"
    },
    {
        "item_id": "A040",
        "headline": "Quantum Computing Breakthroughs: Fault-Tolerant Logical Qubits Demonstrated on Silicon Chips",
        "topic": "Hardware/Chips",
        "publish_date": "2026-03-01",
        "summary_text": "Quantum physics researchers achieved fault-tolerant quantum error correction by encoding logical qubits across silicon-based spin qubit arrays. The demonstration exhibits exponential suppression of physical phase and bit-flip errors as error-correcting codes expand. The milestone demonstrates the viability of leveraging existing semiconductor manufacturing cleanrooms for quantum computing.",
        "tags": "hardware;chips;quantum-computing;qubits;semiconductor;silicon;physics"
    },

    # 9. Programming Languages (A041 - A045)
    {
        "item_id": "A041",
        "headline": "Rust 2026 Edition Solidifies Memory Safety Without Garbage Collection in Systems Programming",
        "topic": "Programming Languages",
        "publish_date": "2026-02-10",
        "summary_text": "The Rust community released the 2026 edition, introducing ergonomic borrow checker refinements and standardized async trait declarations. Systems engineers increasingly replace legacy C++ components with Rust to guarantee memory safety and eliminate null pointer dereferences. The language ecosystem continues its rapid expansion across operating system kernels and web infrastructure.",
        "tags": "programming-languages;rust;systems-programming;memory-safety;concurrency;tooling"
    },
    {
        "item_id": "A042",
        "headline": "Go 1.26 Enhances Goroutine Scheduler Efficiency and Concurrency Memory Footprint",
        "topic": "Programming Languages",
        "publish_date": "2026-03-16",
        "summary_text": "The Go engineering team released Go 1.26 featuring a redesigned runtime scheduler that optimizes work-stealing across heterogeneous multi-core CPUs. Memory consumption per idle goroutine has been halved, allowing distributed web servers to sustain millions of simultaneous TCP connections. Microbenchmarks reveal reduced garbage collector pause times during intensive memory allocations.",
        "tags": "programming-languages;golang;concurrency;goroutines;runtime;performance;backend"
    },
    {
        "item_id": "A043",
        "headline": "Python 3.13 JIT Compiler Delivers Significant Performance Leaps for Numerical Workloads",
        "topic": "Programming Languages",
        "publish_date": "2026-01-18",
        "summary_text": "Python 3.13 introduces an experimental copy-and-patch Just-In-Time compiler alongside continued improvements to free-threaded execution without the Global Interpreter Lock (GIL). Numerical data science and machine learning libraries observe dramatic speedups on tight loops and CPU-bound algorithms. Developers applaud the evolution of Python towards high-performance enterprise systems.",
        "tags": "programming-languages;python;jit;gil;performance;data-science;compiler"
    },
    {
        "item_id": "A044",
        "headline": "Zig Emerges as a Nimble Alternative for Low-Level C Interoperability and Toolchain Ergonomics",
        "topic": "Programming Languages",
        "publish_date": "2026-04-06",
        "summary_text": "The Zig programming language has gained significant traction as a seamless drop-in replacement for traditional C compilers and build systems. With its compile-time code execution (comptime) and explicit memory allocator control, Zig avoids hidden control flow and runtime bloat. Developers use Zig to cross-compile complex multi-platform native libraries with zero external dependencies.",
        "tags": "programming-languages;zig;c;systems-programming;comptime;toolchain;memory-management"
    },
    {
        "item_id": "A045",
        "headline": "Modern C++26 Feature Preview: Reflection, Pattern Matching, and Contracts for Robust Software",
        "topic": "Programming Languages",
        "publish_date": "2026-02-28",
        "summary_text": "ISO C++ standards committee meetings finalized key technical proposals for C++26, including compile-time static reflection and declarative pattern matching. The integration of contract-based programming provides formal verification hooks for critical system invariants. Game engine programmers and financial latency engineers celebrate modern metaprogramming expressive power.",
        "tags": "programming-languages;cpp;systems-programming;metaprogramming;contracts;performance"
    }
]


# User profile taste preferences:
# Dict of user_id -> preferred topics (high ratings 4-5) and secondary/low topics
USER_PROFILES = {
    "U01": {"fav": ["Cybersecurity", "Linux"], "dislike": ["Web Development", "Mobile", "Cloud/DevOps"]},
    "U02": {"fav": ["Linux", "Cloud/DevOps", "Programming Languages"], "dislike": ["Mobile", "AI/ML"]},
    "U03": {"fav": ["Cybersecurity", "Linux", "Hardware/Chips"], "dislike": ["Web Development", "Mobile"]},
    "U04": {"fav": ["Open-Source LLMs", "AI/ML", "Hardware/Chips"], "dislike": ["Web Development", "Mobile"]},
    "U05": {"fav": ["Open-Source LLMs", "AI/ML", "Cloud/DevOps"], "dislike": ["Mobile", "Hardware/Chips"]},
    "U06": {"fav": ["AI/ML", "Open-Source LLMs", "Programming Languages"], "dislike": ["Cybersecurity", "Web Development"]},
    "U07": {"fav": ["Web Development", "Programming Languages", "Cloud/DevOps"], "dislike": ["Hardware/Chips", "Cybersecurity"]},
    "U08": {"fav": ["Web Development", "Mobile"], "dislike": ["Cybersecurity", "Linux", "Hardware/Chips"]},
    "U09": {"fav": ["Mobile", "Hardware/Chips", "Linux"], "dislike": ["Web Development", "Cloud/DevOps"]},
    "U10": {"fav": ["Cloud/DevOps", "Linux", "Web Development"], "dislike": ["Hardware/Chips", "Mobile"]},
    "U11": {"fav": ["Hardware/Chips", "Programming Languages", "AI/ML"], "dislike": ["Web Development", "Mobile"]},
    "U12": {"fav": ["Programming Languages", "Linux", "Cybersecurity"], "dislike": ["Web Development", "Mobile"]},
    "U13": {"fav": ["Cloud/DevOps", "Open-Source LLMs", "Cybersecurity"], "dislike": ["Mobile", "Web Development"]},
    "U14": {"fav": ["Open-Source LLMs", "Web Development", "Mobile"], "dislike": ["Cybersecurity", "Hardware/Chips"]}
}


def generate_ratings(items_df):
    """Generate realistic ratings DataFrame meeting all project constraints."""
    ratings_records = []
    
    # Map item_id to topic
    item_topics = dict(zip(items_df["item_id"], items_df["topic"]))
    all_item_ids = list(items_df["item_id"])
    
    # Track item ratings count to ensure >= 3 for every item
    item_rating_counts = {item_id: 0 for item_id in all_item_ids}
    
    users = [f"U{i:02d}" for i in range(1, 15)]
    
    # Target 18-22 ratings per user (average 20 ratings -> 280 total ratings out of 630 -> 55.6% sparsity)
    # Ensure every user has between 16 and 24 ratings.
    
    user_item_assigned = {u: set() for u in users}
    
    # First, satisfy U01 specific constraint and user taste profiles
    for user_id in users:
        prefs = USER_PROFILES[user_id]
        fav_topics = prefs["fav"]
        dislike_topics = prefs["dislike"]
        
        # Pick 19-21 items per user
        n_items = random.randint(18, 22)
        
        # Find candidate items
        fav_items = [iid for iid in all_item_ids if item_topics[iid] in fav_topics]
        other_items = [iid for iid in all_item_ids if item_topics[iid] not in fav_topics]
        
        # Select about 65-75% from favorite topics, rest from other
        n_fav = min(len(fav_items), int(n_items * 0.65))
        n_other = n_items - n_fav
        
        selected_fav = random.sample(fav_items, n_fav)
        selected_other = random.sample(other_items, min(len(other_items), n_other))
        
        chosen_items = selected_fav + selected_other
        random.shuffle(chosen_items)
        
        for item_id in chosen_items:
            user_item_assigned[user_id].add(item_id)
            item_rating_counts[item_id] += 1
            
            topic = item_topics[item_id]
            # U01: 4-5 on Cybersecurity and Linux, 1-3 on others
            if user_id == "U01":
                if topic in ["Cybersecurity", "Linux"]:
                    rating = random.choice([4, 5, 5])
                else:
                    rating = random.choice([1, 2, 3])
            else:
                if topic in fav_topics:
                    rating = random.choices([4, 5, 3], weights=[0.45, 0.45, 0.10])[0]
                elif topic in dislike_topics:
                    rating = random.choices([1, 2, 3], weights=[0.45, 0.40, 0.15])[0]
                else:
                    rating = random.choices([2, 3, 4], weights=[0.30, 0.45, 0.25])[0]
            
            # Read time loosely correlated with rating (30 to 600 seconds)
            # base: rating * 90 + noise
            base_time = rating * 95
            noise = random.randint(-40, 60)
            read_time = max(35, min(590, base_time + noise))
            
            ratings_records.append({
                "user_id": user_id,
                "item_id": item_id,
                "rating": rating,
                "read_time_sec": read_time
            })
            
    # Check if any item has < 3 ratings, and add to users who have < 24 ratings
    for item_id, count in item_rating_counts.items():
        while count < 3:
            # Find a user who hasn't rated this item and has < 24 ratings
            candidate_users = [u for u in users if item_id not in user_item_assigned[u] and len(user_item_assigned[u]) < 24]
            if not candidate_users:
                candidate_users = [u for u in users if item_id not in user_item_assigned[u]]
            target_user = random.choice(candidate_users)
            user_item_assigned[target_user].add(item_id)
            count += 1
            item_rating_counts[item_id] = count
            
            topic = item_topics[item_id]
            prefs = USER_PROFILES[target_user]
            if target_user == "U01":
                rating = random.choice([4, 5]) if topic in ["Cybersecurity", "Linux"] else random.choice([1, 2, 3])
            elif topic in prefs["fav"]:
                rating = random.choice([4, 5])
            elif topic in prefs["dislike"]:
                rating = random.choice([1, 2])
            else:
                rating = random.choice([2, 3, 4])
                
            base_time = rating * 95
            noise = random.randint(-40, 60)
            read_time = max(35, min(590, base_time + noise))
            
            ratings_records.append({
                "user_id": target_user,
                "item_id": item_id,
                "rating": rating,
                "read_time_sec": read_time
            })
            
    # Sort by user_id and item_id
    ratings_df = pd.DataFrame(ratings_records).drop_duplicates(subset=["user_id", "item_id"])
    ratings_df = ratings_df.sort_values(by=["user_id", "item_id"]).reset_index(drop=True)
    return ratings_df


def validate_dataset(items_df, ratings_df):
    """Validate dataset according to all mini-project constraints."""
    print("=== VALIDATING DATASET ===")
    
    # 1. Items checks
    assert len(items_df) == 45, f"Expected 45 items, got {len(items_df)}"
    expected_cols = {"item_id", "headline", "topic", "publish_date", "summary_text", "tags"}
    assert set(items_df.columns) == expected_cols, f"Items columns mismatch: {items_df.columns}"
    
    # 9 topics x 5
    topic_counts = items_df["topic"].value_counts()
    assert len(topic_counts) == 9, f"Expected 9 topics, got {len(topic_counts)}"
    for topic, count in topic_counts.items():
        assert count == 5, f"Topic '{topic}' has {count} items, expected 5"
    print(f"[PASS] Items: 45 rows across 9 topics (5 articles each).")
    
    # Tags check
    for tags in items_df["tags"]:
        assert ";" in tags, f"Tags should be semicolon-separated: {tags}"
        assert tags.islower(), f"Tags must be lowercase: {tags}"
    print(f"[PASS] Tags formatting verified (lowercase, semicolon-separated).")
    
    # 2. Ratings checks
    expected_rating_cols = {"user_id", "item_id", "rating", "read_time_sec"}
    assert set(ratings_df.columns) == expected_rating_cols, f"Ratings columns mismatch: {ratings_df.columns}"
    
    # Users check
    users = sorted(ratings_df["user_id"].unique())
    assert len(users) == 14, f"Expected 14 users U01-U14, got {len(users)}"
    assert users[0] == "U01" and users[-1] == "U14", f"User IDs mismatch: {users}"
    print(f"[PASS] Users: 14 unique users U01 to U14.")
    
    # Rating range and read_time_sec range
    assert ratings_df["rating"].between(1, 5).all(), "Ratings must be between 1 and 5"
    assert ratings_df["read_time_sec"].between(30, 600).all(), "read_time_sec must be 30-600"
    print(f"[PASS] Ratings (1-5) and read_time_sec (30-600) ranges verified.")
    
    # Per-user counts
    user_counts = ratings_df["user_id"].value_counts()
    assert (user_counts >= 6).all(), f"Each user must have >= 6 ratings: {user_counts.min()}"
    print(f"[PASS] User ratings counts: min={user_counts.min()}, max={user_counts.max()}, mean={user_counts.mean():.1f}")
    
    # Item ratings count >= 3
    item_counts = ratings_df["item_id"].value_counts()
    assert len(item_counts) == 45, f"Not all items rated! Rated items: {len(item_counts)}"
    assert (item_counts >= 3).all(), f"Each item must have >= 3 ratings: min was {item_counts.min()}"
    print(f"[PASS] Every item has >= 3 ratings (min={item_counts.min()}, max={item_counts.max()}).")
    
    # Sparsity
    total_possible = 14 * 45
    total_ratings = len(ratings_df)
    sparsity = (1.0 - (total_ratings / total_possible)) * 100.0
    print(f"[INFO] Total ratings: {total_ratings} / {total_possible}")
    print(f"[INFO] Measured Sparsity: {sparsity:.2f}% (Required: 40-65%)")
    assert 40.0 <= sparsity <= 65.0, f"Sparsity {sparsity:.2f}% outside [40%, 65%]"
    print(f"[PASS] Sparsity is within required [40%, 65%] range.")
    
    # Correlation between rating and read_time_sec
    corr = ratings_df["rating"].corr(ratings_df["read_time_sec"])
    print(f"[INFO] Correlation (rating vs read_time_sec): {corr:.3f}")
    assert corr > 0.4, f"Correlation between rating and read_time_sec should be positive and significant, got {corr}"
    print(f"[PASS] Rating and read_time_sec are positively correlated.")
    
    # U01 taste check
    u01_df = ratings_df[ratings_df["user_id"] == "U01"].merge(items_df, on="item_id")
    u01_sec_linux = u01_df[u01_df["topic"].isin(["Cybersecurity", "Linux"])]
    u01_other = u01_df[~u01_df["topic"].isin(["Cybersecurity", "Linux"])]
    print(f"[INFO] U01 Sec/Linux ratings: {list(u01_sec_linux['rating'])}, mean={u01_sec_linux['rating'].mean():.2f}")
    print(f"[INFO] U01 other ratings mean: {u01_other['rating'].mean():.2f}")
    assert (u01_sec_linux["rating"] >= 4).all(), "U01 must rate Cybersecurity and Linux 4-5"
    assert (u01_other["rating"] <= 3).all(), "U01 must rate other topics 1-3"
    print(f"[PASS] U01 taste profile confirmed (4-5 on Cybersecurity/Linux, 1-3 on others).")
    
    print("=== ALL DATASET VALIDATION CHECKS PASSED ===")
    return True


if __name__ == "__main__":
    items_df = pd.DataFrame(ITEMS_DATA)
    items_path = BASE_DIR / "items.csv"
    items_df.to_csv(items_path, index=False)
    print(f"Saved {items_path} ({len(items_df)} rows)")
    
    ratings_df = generate_ratings(items_df)
    ratings_path = BASE_DIR / "ratings.csv"
    ratings_df.to_csv(ratings_path, index=False)
    print(f"Saved {ratings_path} ({len(ratings_df)} rows)")
    
    validate_dataset(items_df, ratings_df)
