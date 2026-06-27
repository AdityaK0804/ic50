"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  Cpu,
  Layers,
  Search,
  Sparkles,
  BookOpen,
  Database,
  History,
  AlertCircle,
  ChevronRight,
  RefreshCw,
  Sun,
  Moon,
  FlaskConical,
  ExternalLink,
} from "lucide-react";

// Import pre-trained model weights exported from Keras
import modelWeightsRaw from "./model_weights.json";

// Type assertions for weights
const modelWeights = modelWeightsRaw as {
  dense_0_kernel: number[][];
  dense_0_bias: number[];
  bn_1_scale: number[];
  bn_1_offset: number[];
  dense_3_kernel: number[][];
  dense_3_bias: number[];
  dense_4_kernel: number[][];
  dense_4_bias: number[];
};

declare global {
  interface Window {
    initRDKitModule?: () => Promise<any>;
  }
}

// Types for Prediction
interface PredictionResult {
  smiles: string;
  pIC50: number;
  ic50_um: number;
  fingerprint: number[];
}

// Types for History Logs
interface LogEntry {
  id: number;
  smiles: string;
  pic50_pred: number;
  ic50_um_pred: number;
  timestamp: string;
}

// Preset SMILES for easy testing
const PRESET_SMILES = [
  {
    name: "Nifurtimox (Active Drug)",
    smiles: "CC1CCN(N1)N=CC2=CC=C(O2)[N+](=O)[O-]",
  },
  {
    name: "Benznidazole (Active Drug)",
    smiles: "C1=CC=C(C=C1)CNCC2=CN(C=N2)[N+](=O)[O-]",
  },
  {
    name: "High Potency Candidate",
    smiles: "COC1=CC=CC=C1C2=CC=C(C=C2)S(=O)(=O)NCC3=CC=C(C=C3)F",
  },
];

export default function Home() {
  const [activeTab, setActiveTab] = useState<"predict" | "dashboard">("predict");
  const [smilesInput, setSmilesInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [darkMode, setDarkMode] = useState(false);
  const [rdkitInstance, setRdkitInstance] = useState<any>(null);
  const [rdkitStatus, setRdkitStatus] = useState("Loading chemistry module...");

  // Initialize and check dark mode
  useEffect(() => {
    const isDark =
      localStorage.getItem("theme") === "dark" ||
      (!("theme" in localStorage) &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);
    setDarkMode(isDark);
    if (isDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }

    // Load logs from localStorage
    const savedLogs = localStorage.getItem("ic50_forge_logs");
    if (savedLogs) {
      try {
        setLogs(JSON.parse(savedLogs));
      } catch (e) {
        console.error("Failed to load local logs:", e);
      }
    }
  }, []);

  // Load RDKit WebAssembly via CDN dynamically
  useEffect(() => {
    if (typeof window === "undefined") return;
    
    // Check if script is already added
    const existingScript = document.getElementById("rdkit-wasm-script");
    if (existingScript) {
      if ((window as any).rdkit) {
        setRdkitInstance((window as any).rdkit);
        setRdkitStatus("Chemistry module ready");
      }
      return;
    }

    const script = document.createElement("script");
    script.id = "rdkit-wasm-script";
    script.src = "https://unpkg.com/@rdkit/rdkit/dist/RDKit_minimal.js";
    script.async = true;
    script.onload = () => {
      if (window.initRDKitModule) {
        window.initRDKitModule()
          .then((module: any) => {
            console.log("RDKit WASM loaded successfully!");
            (window as any).rdkit = module;
            setRdkitInstance(module);
            setRdkitStatus("Chemistry module ready");
          })
          .catch((err: any) => {
            console.error("Failed to initialize RDKit WASM:", err);
            setRdkitStatus("Failed to load chemistry module");
          });
      }
    };
    script.onerror = () => {
      setRdkitStatus("Failed to download chemistry module script");
    };
    document.body.appendChild(script);
  }, []);

  const toggleDarkMode = () => {
    const nextDark = !darkMode;
    setDarkMode(nextDark);
    if (nextDark) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  };

  // Model Inference Forward Pass in JavaScript
  const runNeuralNetwork = (fp: number[]): number => {
    // Layer 0: Dense (1024 -> 256)
    const w0 = modelWeights.dense_0_kernel;
    const b0 = modelWeights.dense_0_bias;
    const scale = modelWeights.bn_1_scale;
    const offset = modelWeights.bn_1_offset;

    const z0 = new Array(256).fill(0);
    for (let j = 0; j < 256; j++) {
      let sum = 0;
      for (let i = 0; i < 1024; i++) {
        sum += fp[i] * w0[i][j];
      }
      z0[j] = sum + b0[j];
    }

    // Layer 1: ReLU + Batch Normalization
    const a0 = new Array(256).fill(0);
    for (let j = 0; j < 256; j++) {
      const relu = Math.max(0, z0[j]);
      a0[j] = relu * scale[j] + offset[j];
    }

    // Layer 3: Dense (256 -> 64)
    const w3 = modelWeights.dense_3_kernel;
    const b3 = modelWeights.dense_3_bias;
    const z3 = new Array(64).fill(0);
    for (let j = 0; j < 64; j++) {
      let sum = 0;
      for (let i = 0; i < 256; i++) {
        sum += a0[i] * w3[i][j];
      }
      z3[j] = sum + b3[j];
    }

    // Activation: ReLU
    const a3 = z3.map((val) => Math.max(0, val));

    // Layer 4: Dense (64 -> 1, Linear)
    const w4 = modelWeights.dense_4_kernel;
    const b4 = modelWeights.dense_4_bias;
    let y = 0;
    for (let i = 0; i < 64; i++) {
      y += a3[i] * w4[i][0];
    }
    return y + b4[0];
  };

  const handlePredict = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!smilesInput.trim()) {
      setErrorMsg("Please enter a valid SMILES string.");
      return;
    }

    if (!rdkitInstance) {
      setErrorMsg("The chemical WebAssembly module is still loading. Please wait.");
      return;
    }

    setIsLoading(true);
    setErrorMsg("");
    setResult(null);

    // Run prediction client-side in a small timeout to let loading state show
    setTimeout(() => {
      let mol = null;
      try {
        const smiles = smilesInput.trim();
        mol = rdkitInstance.get_mol(smiles);
        
        if (!mol) {
          setErrorMsg("Invalid SMILES string. RDKit failed to parse the molecule.");
          setIsLoading(false);
          return;
        }

        // Generate 1024-bit Morgan Fingerprint radius 2
        const fpBinaryText = mol.get_morgan_fp_as_binary_text(2, 1024) as string;

        if (!fpBinaryText || fpBinaryText.length !== 1024) {
          setErrorMsg("Failed to generate molecular fingerprint descriptor.");
          setIsLoading(false);
          return;
        }

        // Convert bit string to number array
        const fpArray = Array.from(fpBinaryText).map((char: string) => parseInt(char, 10));

        // Predict pIC50 using pre-trained weights
        const pic50Pred = runNeuralNetwork(fpArray);
        
        // Convert pIC50 back to micromolar: 10^(6 - pIC50)
        const ic50UmPred = Math.pow(10, 6 - pic50Pred);

        const predictionData: PredictionResult = {
          smiles,
          pIC50: pic50Pred,
          ic50_um: ic50UmPred,
          fingerprint: fpArray,
        };

        setResult(predictionData);

        // Log prediction to LocalStorage database logs
        const newLogEntry: LogEntry = {
          id: Date.now(),
          smiles: smiles,
          pic50_pred: pic50Pred,
          ic50_um_pred: ic50UmPred,
          timestamp: new Date().toISOString().replace("T", " ").substring(0, 19),
        };

        const updatedLogs = [newLogEntry, ...logs];
        setLogs(updatedLogs);
        localStorage.setItem("ic50_forge_logs", JSON.stringify(updatedLogs));

      } catch (err) {
        setErrorMsg("An error occurred during client-side chemical calculation.");
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    }, 50);
  };

  const handlePresetClick = (smiles: string) => {
    setSmilesInput(smiles);
    setErrorMsg("");
  };

  const handleNavClick = (e: React.MouseEvent, anchorId: string) => {
    e.preventDefault();
    setActiveTab("predict");
    setTimeout(() => {
      const element = document.getElementById(anchorId);
      if (element) {
        element.scrollIntoView({ behavior: "smooth" });
      }
    }, 100);
  };

  const clearLogs = () => {
    if (confirm("Are you sure you want to clear your prediction history?")) {
      setLogs([]);
      localStorage.removeItem("ic50_forge_logs");
    }
  };

  const getPotencyBadge = (pic50: number) => {
    if (pic50 >= 6.0) {
      return (
        <span className="px-3 py-1.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 shadow-sm">
          High Potency (pIC50 ≥ 6)
        </span>
      );
    } else if (pic50 >= 5.0) {
      return (
        <span className="px-3 py-1.5 rounded-full text-xs font-semibold bg-cyan-500/10 text-cyan-500 border border-cyan-500/20 shadow-sm">
          {"Medium Potency (5 ≤ pIC50 < 6)"}
        </span>
      );
    } else {
      return (
        <span className="px-3 py-1.5 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-500 border border-rose-500/20 shadow-sm">
          {"Low Potency (pIC50 < 5)"}
        </span>
      );
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-apple-dark text-gray-900 dark:text-gray-100 flex flex-col font-sans selection:bg-emerald-500/30 transition-colors duration-300">
      
      {/* Sticky Glassmorphic Navbar */}
      <nav className="glass-navbar sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3 cursor-pointer" onClick={(e) => handleNavClick(e, "hero")}>
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-emerald-400 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/20 border border-emerald-400/20">
            <FlaskConical className="w-5 h-5 text-white" />
          </div>
          <span className="text-xl font-bold tracking-tight">
            IC50 <span className="bg-gradient-to-r from-emerald-500 to-cyan-500 bg-clip-text text-transparent">FORGE</span>
          </span>
        </div>

        <div className="flex items-center space-x-8">
          <a
            href="#features"
            onClick={(e) => handleNavClick(e, "features")}
            className="text-sm font-medium text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            Features
          </a>
          <a
            href="#how-it-works"
            onClick={(e) => handleNavClick(e, "how-it-works")}
            className="text-sm font-medium text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            How it Works
          </a>

          {/* Apple-style Light/Dark Toggle */}
          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-full hover:bg-gray-200/50 dark:hover:bg-gray-800/50 transition-colors text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white cursor-pointer"
            aria-label="Toggle Theme"
          >
            {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>

          {/* Rounded Biotech Green/Blue Button with White Text */}
          <motion.button
            whileHover={{ scale: 1.05, boxShadow: "0 0 20px rgba(16, 185, 129, 0.4)" }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setActiveTab(activeTab === "predict" ? "dashboard" : "predict")}
            className="px-6 py-2.5 rounded-full text-sm font-semibold bg-gradient-to-r from-emerald-500 to-cyan-600 hover:from-emerald-600 hover:to-cyan-700 text-white shadow-md flex items-center space-x-2 transition-colors cursor-pointer border border-emerald-400/20"
          >
            {activeTab === "predict" ? (
              <>
                <History className="w-4 h-4" />
                <span>Dashboard</span>
              </>
            ) : (
              <>
                <Activity className="w-4 h-4" />
                <span>Predictor</span>
              </>
            )}
          </motion.button>
        </div>
      </nav>

      <main className="flex-grow max-w-6xl mx-auto w-full px-6 py-12">
        
        {/* Connection status indicator for WebAssembly module */}
        <div className="flex justify-end mb-4">
          <span className="flex items-center space-x-2 text-xs font-mono text-gray-400">
            <span className={`w-2 h-2 rounded-full ${rdkitInstance ? "bg-emerald-500" : "bg-amber-500 animate-pulse"}`} />
            <span>{rdkitStatus}</span>
          </span>
        </div>

        <AnimatePresence mode="wait">
          {activeTab === "predict" ? (
            <motion.div
              key="predict-tab"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              className="space-y-24"
            >
              {/* Hero Prediction Section */}
              <section id="hero" className="flex flex-col items-center text-center max-w-3xl mx-auto pt-4 scroll-mt-24">
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.1, duration: 0.5 }}
                  className="space-y-4 mb-10"
                >
                  <span className="px-3.5 py-1.5 rounded-full text-xs font-semibold bg-emerald-500/10 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20 tracking-wide uppercase">
                    Edge computing — 100% on Vercel
                  </span>
                  <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight leading-tight">
                    Predict Chagas Disease <br />
                    <span className="bg-gradient-to-r from-emerald-400 to-cyan-500 bg-clip-text text-transparent">
                      Drug Potency Instantly
                    </span>
                  </h1>
                  <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl">
                    Enter a molecular SMILES representation. Our WebAssembly chemistry pipeline and JavaScript neural network compute the compound's \(pIC_{50}\) instantly in your browser.
                  </p>
                </motion.div>

                {/* Centralized Glassmorphic Input Card */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2, duration: 0.6 }}
                  className="w-full glass-card rounded-3xl p-6 md:p-8 space-y-6 text-left hover:border-emerald-500/30 transition-all duration-300"
                >
                  <form onSubmit={handlePredict} className="space-y-4">
                    <div className="flex items-center justify-between">
                      <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300">
                        Molecular SMILES String
                      </label>
                      <span className="text-xs text-gray-400 flex items-center gap-1">
                        <FlaskConical className="w-3.5 h-3.5" /> input representation
                      </span>
                    </div>
                    <div className="relative flex flex-col md:flex-row gap-3">
                      <div className="relative flex-grow">
                        <input
                          type="text"
                          value={smilesInput}
                          onChange={(e) => setSmilesInput(e.target.value)}
                          placeholder="Paste SMILES (e.g. CC1CCN(N1)N=CC2=CC=C(O2)[N+](=O)[O-]...) or click preset"
                          className="w-full pl-12 pr-4 py-4 rounded-2xl bg-white/70 dark:bg-zinc-900/50 border border-gray-200 dark:border-zinc-800 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-transparent transition-all shadow-inner text-sm md:text-base font-mono"
                        />
                        <Search className="w-5 h-5 text-gray-400 absolute left-4 top-1/2 transform -translate-y-1/2" />
                      </div>
                      <motion.button
                        whileHover={{ scale: 1.02, boxShadow: "0 0 15px rgba(16, 185, 129, 0.3)" }}
                        whileTap={{ scale: 0.98 }}
                        type="submit"
                        disabled={isLoading}
                        className="px-8 py-4 rounded-2xl bg-gradient-to-r from-emerald-500 to-cyan-600 hover:from-emerald-600 hover:to-cyan-700 text-white font-semibold transition-colors flex items-center justify-center space-x-2 shadow-lg shadow-emerald-500/10 cursor-pointer disabled:opacity-75"
                      >
                        {isLoading ? (
                          <RefreshCw className="w-5 h-5 animate-spin" />
                        ) : (
                          <Sparkles className="w-5 h-5 text-emerald-200" />
                        )}
                        <span>{isLoading ? "Predicting..." : "Predict IC50"}</span>
                      </motion.button>
                    </div>
                  </form>

                  {/* Preset SMILES Quick Tests */}
                  <div className="flex flex-wrap items-center gap-3 pt-2">
                    <span className="text-xs font-semibold text-gray-400">Presets:</span>
                    {PRESET_SMILES.map((preset) => (
                      <motion.button
                        whileHover={{ scale: 1.03, borderColor: "#10b981" }}
                        whileTap={{ scale: 0.97 }}
                        key={preset.name}
                        onClick={() => handlePresetClick(preset.smiles)}
                        className="px-3.5 py-1.5 rounded-full text-xs font-medium bg-white dark:bg-zinc-900/50 hover:bg-gray-100 dark:hover:bg-zinc-800/80 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-zinc-800 shadow-sm transition-all cursor-pointer"
                      >
                        {preset.name}
                      </motion.button>
                    ))}
                  </div>

                  {/* Errors */}
                  {errorMsg && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-start space-x-3 text-rose-600 dark:text-rose-400 text-sm"
                    >
                      <AlertCircle className="w-5 h-5 text-rose-500 flex-shrink-0 mt-0.5" />
                      <span>{errorMsg}</span>
                    </motion.div>
                  )}

                  {/* Prediction Results Display */}
                  {result && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.98 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.4 }}
                      className="pt-6 border-t border-gray-200/50 dark:border-zinc-800 space-y-6"
                    >
                      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                        <div>
                          <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider block">
                            Target Molecule SMILES
                          </span>
                          <span className="text-sm font-mono text-gray-700 dark:text-gray-300 break-all select-all">
                            {result.smiles}
                          </span>
                        </div>
                        <div className="flex-shrink-0">
                          {getPotencyBadge(result.pIC50)}
                        </div>
                      </div>

                      {/* Display Numbers */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-white/40 dark:bg-zinc-900/20 p-6 rounded-2xl border border-white/60 dark:border-zinc-800/50">
                        <div className="space-y-1">
                          <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                            Predicted pIC50 (-log₁₀ M)
                          </span>
                          <div className="text-4xl md:text-5xl font-black text-gray-900 dark:text-white font-mono tracking-tight">
                            {result.pIC50.toFixed(4)}
                          </div>
                          <p className="text-xs text-gray-400">
                            Higher values indicate higher potency.
                          </p>
                        </div>

                        <div className="space-y-1">
                          <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                            Predicted IC50 (µM)
                          </span>
                          <div className="text-4xl md:text-5xl font-black text-emerald-500 dark:text-emerald-400 font-mono tracking-tight">
                            {result.ic50_um.toFixed(4)} <span className="text-lg font-semibold text-gray-400">µM</span>
                          </div>
                          <p className="text-xs text-gray-400">
                            Inhibition concentration required for 50% response.
                          </p>
                        </div>
                      </div>

                      {/* Morgan Fingerprint Visualizer */}
                      {result.fingerprint && (
                        <div className="space-y-3">
                          <div className="flex justify-between items-center">
                            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider block">
                              1024-bit Morgan Fingerprint Vector
                            </span>
                            <span className="text-xs font-medium text-emerald-500 dark:text-emerald-400 font-mono">
                              Active Bits: {result.fingerprint.filter((bit) => bit === 1).length} / 1024
                            </span>
                          </div>
                          
                          {/* Grid layout for bits */}
                          <div className="grid grid-cols-32 gap-0.5 p-3 bg-gray-950 rounded-2xl overflow-hidden border border-gray-900 shadow-inner">
                            {result.fingerprint.map((bit, idx) => (
                              <div
                                key={idx}
                                title={`Bit ${idx}: ${bit}`}
                                className={`aspect-square w-full rounded-sm transition-all duration-300 ${
                                  bit === 1
                                    ? "bg-emerald-400 shadow-[0_0_6px_#10b981]"
                                    : "bg-gray-800/80 hover:bg-gray-700/60"
                                }`}
                              />
                            ))}
                          </div>
                        </div>
                      )}
                    </motion.div>
                  )}
                </motion.div>
              </section>

              {/* Features Section */}
              <section id="features" className="scroll-mt-24 space-y-12">
                <div className="text-center max-w-xl mx-auto space-y-2">
                  <h2 className="text-3xl font-extrabold tracking-tight">
                    Platform Features
                  </h2>
                  <p className="text-gray-500 dark:text-gray-400 text-sm">
                    Highly optimized chemical descriptors integrated with neural networks for state-of-the-art client-side predictions.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                  {[
                    {
                      icon: <Layers className="w-6 h-6 text-emerald-500 dark:text-emerald-400" />,
                      title: "WebAssembly RDKit",
                      description:
                        "Loads chemical calculations directly in the browser using the official RDKit WebAssembly module, generating identical 1024-bit Morgan fingerprints.",
                    },
                    {
                      icon: <Cpu className="w-6 h-6 text-emerald-500 dark:text-emerald-400" />,
                      title: "Feedforward NN",
                      description:
                        "Runs predictions in less than 1ms using JavaScript matrix calculations powered by pre-compiled weights and biases exported from our Keras model.",
                    },
                    {
                      icon: <Database className="w-6 h-6 text-emerald-500 dark:text-emerald-400" />,
                      title: "Persistent Local Logs",
                      description:
                        "Predictions are securely written to the browser's local storage database, maintaining your history between restarts without database outages.",
                    },
                  ].map((feat, index) => (
                    <motion.div
                      key={feat.title}
                      initial={{ opacity: 0, y: 15 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true, margin: "-100px" }}
                      transition={{ delay: index * 0.1, duration: 0.5 }}
                      whileHover={{ 
                        y: -6, 
                        boxShadow: "0 12px 30px rgba(16, 185, 129, 0.08)",
                        borderColor: "rgba(16, 185, 129, 0.3)"
                      }}
                      className="bg-white dark:bg-zinc-900/40 p-6 rounded-3xl border border-gray-150 dark:border-zinc-800/80 shadow-sm space-y-4 transition-colors"
                    >
                      <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20">
                        {feat.icon}
                      </div>
                      <h3 className="text-lg font-bold">{feat.title}</h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">{feat.description}</p>
                    </motion.div>
                  ))}
                </div>
              </section>

              {/* How it Works Section */}
              <section id="how-it-works" className="scroll-mt-24 bg-white dark:bg-zinc-900/20 p-8 md:p-12 rounded-3xl border border-gray-150 dark:border-zinc-800/80 shadow-sm space-y-12">
                <div className="text-center max-w-xl mx-auto space-y-2">
                  <h2 className="text-3xl font-extrabold tracking-tight">
                    Predictive Pipeline
                  </h2>
                  <p className="text-gray-500 dark:text-gray-400 text-sm">
                    How IC50 FORGE maps chemical notations to quantitative bioactivity predictions.
                  </p>
                </div>

                {/* Pipeline Flowchart */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-8 relative">
                  {[
                    {
                      step: "01",
                      title: "SMILES String",
                      desc: "User inputs a compound's SMILES string, representing its molecular structure.",
                    },
                    {
                      step: "02",
                      title: "WASM Descriptor",
                      desc: "RDKit compiles the chemical graph in WebAssembly, outputting a 1024-bit Morgan Fingerprint.",
                    },
                    {
                      step: "03",
                      title: "JS Inference",
                      desc: "Calculates the neural network forward pass dynamically inside the page component context.",
                    },
                    {
                      step: "04",
                      title: "Potency Value",
                      desc: "Outputs computed pIC50 and calculates the corresponding value in micromolar (µM).",
                    },
                  ].map((pipe, index) => (
                    <motion.div
                      key={pipe.title}
                      initial={{ opacity: 0, x: -10 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true, margin: "-100px" }}
                      transition={{ delay: index * 0.1, duration: 0.5 }}
                      whileHover={{ y: -3 }}
                      className="space-y-3 relative z-10 p-4 rounded-2xl hover:bg-gray-100/50 dark:hover:bg-zinc-800/25 transition-colors"
                    >
                      <div className="text-4xl font-black text-emerald-500/20 dark:text-emerald-500/10 font-mono">
                        {pipe.step}
                      </div>
                      <h4 className="text-base font-bold">{pipe.title}</h4>
                      <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">{pipe.desc}</p>
                    </motion.div>
                  ))}
                </div>
              </section>
            </motion.div>
          ) : (
            /* Dashboard Tab - Prediction History Audit Logs */
            <motion.div
              key="dashboard-tab"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              className="space-y-8 pt-4"
            >
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h1 className="text-3xl font-extrabold tracking-tight">
                    Prediction Logs Dashboard
                  </h1>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    A secure repository logging all successful drug predictions written in the local storage database.
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={clearLogs}
                    className="px-4 py-2.5 rounded-full text-xs font-semibold bg-rose-500/10 hover:bg-rose-500/20 text-rose-600 border border-rose-500/20 shadow-sm flex items-center space-x-1.5 transition-colors cursor-pointer"
                  >
                    <span>Clear Logs</span>
                  </motion.button>
                </div>
              </div>

              {/* logs display */}
              <div className="bg-white dark:bg-zinc-900/40 rounded-3xl border border-gray-200 dark:border-zinc-800 overflow-hidden shadow-sm">
                {logs.length === 0 ? (
                  <div className="py-20 text-center space-y-3">
                    <div className="w-12 h-12 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mx-auto text-emerald-500">
                      <Database className="w-6 h-6" />
                    </div>
                    <p className="font-semibold text-base">No Predictions Logged Yet</p>
                    <p className="text-gray-500 text-sm max-w-sm mx-auto">
                      Submit chemical SMILES strings on the Predictor tab to start logging records.
                    </p>
                    <button
                      onClick={(e) => handleNavClick(e, "hero")}
                      className="mt-2 text-sm font-semibold text-emerald-500 dark:text-emerald-400 hover:underline flex items-center justify-center space-x-1 mx-auto cursor-pointer"
                    >
                      <span>Go to Predictor</span>
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="bg-gray-50/50 dark:bg-zinc-900/50 border-b border-gray-200 dark:border-zinc-800 text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          <th className="py-4 px-6 w-16">ID</th>
                          <th className="py-4 px-6">SMILES Notation</th>
                          <th className="py-4 px-6 w-32 text-right">Predicted pIC50</th>
                          <th className="py-4 px-6 w-36 text-right">Predicted IC50 (µM)</th>
                          <th className="py-4 px-6 w-52 text-right">Timestamp</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100 dark:divide-zinc-800/80 text-sm">
                        {logs.map((log, idx) => (
                          <tr key={log.id} className="hover:bg-gray-50/50 dark:hover:bg-zinc-800/20 transition-colors">
                            <td className="py-4 px-6 font-mono text-gray-400">#{logs.length - idx}</td>
                            <td className="py-4 px-6">
                              <div className="max-w-[400px] md:max-w-[500px] truncate font-mono text-xs text-gray-700 dark:text-gray-300 select-all" title={log.smiles}>
                                {log.smiles}
                              </div>
                            </td>
                            <td className="py-4 px-6 text-right font-mono font-semibold">
                              {log.pic50_pred.toFixed(4)}
                            </td>
                            <td className="py-4 px-6 text-right font-mono font-semibold text-emerald-500 dark:text-emerald-400">
                              {log.ic50_um_pred.toFixed(4)} µM
                            </td>
                            <td className="py-4 px-6 text-right text-xs text-gray-500 font-mono">
                              {log.timestamp}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200/50 dark:border-zinc-800/80 bg-white dark:bg-zinc-950 py-8 mt-12 text-center text-xs text-gray-500 space-y-2 transition-colors">
        <div>
          <strong>IC50 FORGE</strong> — Client-side AI-driven Chagas disease drug potency predictor.
        </div>
        <div className="flex justify-center space-x-6">
          <span>Model: Keras Sequential (DNN) in JS</span>
          <span>Descriptor: 1024-bit Morgan FP in WASM</span>
          <span>Database: HTML5 LocalStorage</span>
        </div>
      </footer>
    </div>
  );
}
