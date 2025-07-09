import "./App.css";
import React, { useState, useRef, useEffect } from "react";
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemText,
  Container,
  Grid,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  TextField,
  Button,
  Box,
  FormControlLabel,
} from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { FormLabel, FormGroup } from "@mui/material";
import { Slider } from "@mui/material";
import Tooltip from "@mui/material/Tooltip";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

const wineTheme = createTheme({
  palette: {
    primary: {
      main: "#7b1e3c", // button color
    },
    background: {
      default: "#fdf6e3",
      paper: "#fbeaea",
    },
    text: {
      primary: "#5b0e2d",
    },
  },
});
const drawerWidth = 240;

const projectionLabels = {
  scores: "Scores",
  concatenated: "Concatenated",
  tic: "TIC",
  tis: "TIS",
  tic_tis: "TIC + TIS",
};

const taskOptionsByFamily = {
  bordeaux: ["Classification"],
  pinot: ["Classification"],
  press: ["Classification"],
  champagne: [
    "Predict Labels",
    "Predict Age",
    "Model Global",
    "Model per Taster",
  ],
};

const shouldShowAdvancedOptions = (wineFamily, task) => {
  const hiddenCombinations = [
    { wineFamily: "champagne", task: "Predict Labels" },
    //     { wineFamily: "champagne", task: "Predict Age" },
    { wineFamily: "champagne", task: "Model Global" },
    { wineFamily: "champagne", task: "Model per taster" },
    // Add more exclusions here if needed
  ];

  return !hiddenCombinations.some(
    (combo) => combo.wineFamily === wineFamily && combo.task === task,
  );
};

function App() {
  const logsEndRef = useRef(null);
  const [selectedMenu, setSelectedMenu] = useState("Dashboard");
  const [logs, setLogs] = useState("");
  const defaultFeatureType = "tic_tis"; // or whatever default you expect
  const [featureType, setFeatureType] = useState("tic_tis");
  const [cvType, setCvType] = useState("LOOPC");
  const [showConfusionMatrix, setShowConfusionMatrix] = useState(false);
  const [wineFamily, setWineFamily] = useState("bordeaux");
  const [task, setTask] = useState("classification");
  const [taskOptions, setTaskOptions] = useState([]);
  const [selectedTask, setSelectedTask] = useState("");
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);
  const [classifier, setClassifier] = useState("RGC");
  const [regressor, setRegressor] = useState("ridge");
  const [numRepeats, setNumRepeats] = useState(10);
  const [chromCap, setChromCap] = useState(35000);
  const [normalize, setNormalize] = useState(true);
  const [syncState, setSyncState] = useState(false);
  const [classByYear, setClassByYear] = useState(false);
  const [region, setRegion] = useState("");
  const [selectedDatasets, setSelectedDatasets] = useState(["bordeaux_oak"]);
  const [plotProjection, setPlotProjection] = useState(false);
  const [projectionMethod, setProjectionMethod] = useState("UMAP");
  const [colorByCountry, setColorByCountry] = useState(false);
  const [projectionDim, setProjectionDim] = useState(2);
  const [nNeighbors, setNNeighbors] = useState(30);
  const [perplexity, setPerplexity] = useState(5); // default for t-SNE
  const [randomState, setRandomState] = useState(42);
  const [invertX, setInvertX] = useState(false);
  const [invertY, setInvertY] = useState(false);
  const [umapData, setUmapData] = useState(null);
  const [labelTargets, setLabelTargets] = useState(["taster"]);
  const [showSampleNames, setShowSampleNames] = useState(false);
  const [groupWines, setGroupWines] = useState(false);
  const [showGlobalFocusHeatmap, setShowGlobalFocusHeatmap] = useState(false);
  const [showTasterFocusHeatmap, setShowTasterFocusHeatmap] = useState(false);
  const [plotR2, setPlotR2] = useState(false);
  const [reduceDims, setReduceDims] = React.useState(false);
  const [reductionMethod, setReductionMethod] = React.useState("pca");
  const [reductionDims, setReductionDims] = React.useState(2);
  const [doClassification, setDoClassification] = useState(false);
  const [showChampPredictedProfiles, setShowChampPredictedProfiles] =
    useState(false);
  //   const [useChampTasterScaling, setUseChampTasterScaling] = useState(false);
  //   const [shuffleLabels, setShuffleLabels] = useState(false);
  //   const [testAverageScores, setTestAverageScores] = useState(false);
  //   const [tasterVsMean, setTasterVsMean] = useState(false);
  const champagnePredictLabelsProjectionOptions = [
    { value: "scores", label: "Class. Scores" },
    { value: "sensory", label: "Sensory Features" },
  ];
  const [showPredPlot, setShowPredPlot] = useState(false);
  const [showAgeHist, setShowAgeHist] = useState(false);
  const isChampagneAgePrediction =
    wineFamily === "champagne" && selectedTask === "Predict Age";
  const isChampagneModelGlobal =
    wineFamily === "champagne" && selectedTask === "Model Global";
  const isChampagneModelPerTaster =
    wineFamily === "champagne" && selectedTask === "Model per Taster";

  const [showChroms, setShowChroms] = useState(false);
  const [rtRange, setRtRange] = useState([0, 30000]);
  const featureToProjectionOptions = {
    tic: [
      { value: "scores", label: "Class. Scores" },
      { value: "tic", label: "TIC" },
    ],
    tis: [
      { value: "scores", label: "Class. Scores" },
      { value: "tis", label: "TIS" },
    ],
    tic_tis: [
      { value: "scores", label: "Class. Scores" },
      { value: "tic_tis", label: "TIC + TIS" },
    ],
  };
  const defaultProjectionOptions =
    featureToProjectionOptions[defaultFeatureType] || [];
  const [projectionSource, setProjectionSource] = useState(
    defaultProjectionOptions[0]?.value || "",
  );
  const [availableUmapOptions, setAvailableUmapOptions] = useState(
    featureToProjectionOptions[featureType],
  );
  const datasetOptions = {
    bordeaux: ["bordeaux_oak"],
    press: [
      "merlot_2021",
      "merlot_2022",
      "merlot_2023",
      "cab_sauv_2021",
      "cab_sauv_2022",
      "cab_sauv_2023",
    ],
    pinot: [
      "pinot_noir_changins",
      "pinot_noir_isvv_lle",
      "pinot_noir_isvv_dllme",
    ],
    champagne: ["heterocyc"],
  };
  const defaultDatasetByScript = {
    bordeaux: ["bordeaux_oak"],
    press: ["merlot_2022"],
    pinot: ["pinot_noir_changins"], // or your actual default for pinot
    champagne: ["heterocyc"],
  };
  const [tasterTests, setTasterTests] = useState([]); // values: "scaling", "shuffle", "average"

  const testOptions = [
    {
      value: "scaling",
      label: "Taster Scaling",
      tooltip:
        "Adds a scaling layer (sensitivity) for each taster during learning",
    },
    {
      value: "shuffle",
      label: "Shuffle Labels",
      tooltip:
        "Randomly reassigns sensory scores of each sample to chromatograms",
    },
    {
      value: "average",
      label: "Average Scores",
      tooltip:
        "Trains and evaluates a model on average sensory scores, excluding individual tasters identity",
    },
    {
      value: "vsmean",
      label: "Taster vs Mean",
      tooltip: "Compares each individual taster with the average of the rest",
    },
    {
      value: "removeavg",
      label: "OHE Subtract Score Avgs",
      tooltip: "Removes average scores across wines for each attribute for each taster before training",
    },
   {
      value: "constantohe",
      label: "Constant OHE ",
      tooltip: "Adds a dummy One-hot encoding vector with all zero",
    },
    {
      value: "plotallr2",
      label: "Compare All Tests",
      tooltip: "Plots R² of each test for comparison",
    },
  ];

  useEffect(() => {
    const firstOption = featureToProjectionOptions[featureType][0];
    if (firstOption) {
      setProjectionSource(firstOption.value);
    }
  }, [featureType]);

  useEffect(() => {
    if (wineFamily) {
      setTaskOptions(taskOptionsByFamily[wineFamily] || []);
      setSelectedTask(""); // reset selection when wineFamily changes
    } else {
      setTaskOptions([]);
      setSelectedTask("");
    }
  }, [wineFamily]);

  useEffect(() => {
    const options = taskOptionsByFamily[wineFamily] || [];
    if (options.length > 0) {
      setSelectedTask(options[0]);
    } else {
      setSelectedTask("");
    }
  }, [wineFamily]);

  useEffect(() => {
    if (wineFamily === "pinot") {
      setRegion("winery");
    } else {
      setRegion(""); // or null
    }
  }, [wineFamily]);

  const outputRef = useRef(null);
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output]);

  useEffect(() => {
    if ( (wineFamily === "bordeaux" || wineFamily === "pinot" || wineFamily === "press") &&
      (selectedTask === "Classification" || selectedTask === "Plot Projection")
    ) {
      setProjectionSource("scores");
    }
  }, [wineFamily, selectedTask]);


  useEffect(() => {
    if (wineFamily === "champagne" && selectedTask === "Predict Labels") {
      setSelectedDatasets(["sensory_scores"]);
      setProjectionSource("sensory");
    }
  }, [wineFamily, selectedTask]);

  useEffect(() => {
    if (wineFamily === "champagne" && selectedTask === "Predict Age") {
      setSelectedDatasets(["heterocyc"]);
      setProjectionSource("sensory");
    }
  }, [wineFamily, selectedTask]);

  useEffect(() => {
    if (
      (isChampagneAgePrediction ||
        isChampagneModelGlobal ||
        isChampagneModelPerTaster) &&
      !regressor
    ) {
      setRegressor("ridge");
    } else if (
      !isChampagneAgePrediction &&
      !isChampagneModelGlobal &&
      !isChampagneModelPerTaster &&
      !classifier
    ) {
      setClassifier("RGC");
    }
  }, [isChampagneAgePrediction, regressor, classifier]);

  useEffect(() => {
    if (
      (wineFamily === "champagne" && selectedTask === "Model Global") ||
      selectedTask === "Model per Taster"
    ) {
      setSelectedDatasets(["heterocyc"]); // or your desired default
      setFeatureType("tic"); // if needed
    }
  }, [wineFamily, selectedTask]);

  useEffect(() => {
    let eventSource;

    if (selectedMenu === "Logs") {
//       eventSource = new EventSource("/logs");
      eventSource = new EventSource("http://localhost:8000/logs");
      eventSource.onmessage = (e) => {
        setLogs((prev) => prev + e.data + "\n");
      };
    }

  return () => {
    if (eventSource) {
      eventSource.close();
    }
  };
}, [selectedMenu]);

  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs]);


  const useChampTasterScaling = tasterTests.includes("scaling");
  const shuffleLabels = tasterTests.includes("shuffle");
  const testAverageScores = tasterTests.includes("average");
  const tasterVsMean = tasterTests.includes("vsmean");
  const plotAllTests = tasterTests.includes("plotallr2");
  const removeAvgScores = tasterTests.includes("removeavg");
  const constantohe = tasterTests.includes("constantohe");

  const run = async () => {
    setLoading(true);
    setOutput("");

    // Build payload first, excluding region by default
    const payload = {
      script_key: wineFamily,
      classifier: isChampagneAgePrediction && doClassification ? classifier : null,
      regressor: isChampagneAgePrediction && doClassification ? null : regressor,
//       classifier:
//         isChampagneAgePrediction ||
//         isChampagneModelGlobal ||
//         isChampagneModelPerTaster
//           ? null
//           : classifier,
//       regressor:
//         isChampagneAgePrediction ||
//         isChampagneModelGlobal ||
//         isChampagneModelPerTaster
//           ? regressor
//           : null,
      feature_type: featureType,
      num_repeats: numRepeats,
      chrom_cap: chromCap,
      normalize,
      sync_state: syncState,
      class_by_year: classByYear,
      selected_datasets: selectedDatasets,
      show_confusion_matrix: showConfusionMatrix,
      plot_projection: plotProjection ? projectionSource : false,
      projection_method: projectionMethod,
      color_by_country: colorByCountry,
      projection_source: projectionSource,
      projection_dim: projectionDim,
      n_neighbors: projectionMethod === "UMAP" ? nNeighbors : null,
      perplexity: projectionMethod === "t-SNE" ? perplexity : null,
      random_state: randomState,
      region: region,
      show_sample_names: showSampleNames,
      show_pred_plot: showPredPlot,
      show_age_histogram: showAgeHist,
      show_chromatograms: showChroms,
      do_classification: doClassification,
      rt_range: { min: rtRange[0], max: rtRange[1] },
      cv_type: cvType,
      invert_x: invertX,
      invert_y: invertY,
      plot_r2: plotR2,
    };

    // ✅ Conditionally add region only for pinot
    if (wineFamily === "pinot") {
      payload.region = region;
    }
    if (wineFamily === "champagne") {
      payload.script_key = {
        "Predict Labels": "champagne_predict_labels",
        "Predict Age": "champagne_predict_age",
        "Model Global": "champagne_global_model",
        "Model per Taster": "champagne_per_taster_model",
      }[selectedTask];
    }
    if (wineFamily === "champagne" && selectedTask === "Predict Labels") {
      payload.label_targets = labelTargets; // ✅ This sends your multiselect to the backend
    }
    if (wineFamily === "champagne" && selectedTask === "Model Global") {
      payload.group_wines = groupWines;
      payload.show_predicted_profiles = showChampPredictedProfiles;
      payload.taster_scaling = useChampTasterScaling;
      payload.shuffle_labels = shuffleLabels;
      payload.test_average_scores = testAverageScores;
      payload.taster_vs_mean = tasterVsMean;
      payload.plot_all_tests = plotAllTests;
      payload.remove_avg_scores = removeAvgScores;
      payload.constant_ohe = constantohe;
      payload.reduce_dims = reduceDims;
      payload.reduction_method = reductionMethod;
      payload.reduction_dims = reductionDims;

    }

    if (wineFamily === "champagne" && selectedTask === "Model per Taster") {
      payload.global_focus_heatmap = showGlobalFocusHeatmap;
      payload.taster_focus_heatmap = showTasterFocusHeatmap;
    }
    const logMessage = (msg) => {
      const timestamp = new Date().toLocaleTimeString();
      setLogs((prev) => [...prev, `[${timestamp}] ${msg}`]);
      console.log(msg);
    };

    // logMessage("Selected labelTargets:", labelTargets);

    // logMessage("Payload:", payload);

    // Send request
//     const res = await fetch("http://localhost:8000/run-script", {
     const res = await fetch("/run-script", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      setOutput((prev) => prev + decoder.decode(value));
    }

    setLoading(false);
  };
  return (
    <ThemeProvider theme={wineTheme}>
      {
        <Box sx={{ display: "flex" }}>
          <AppBar
            position="fixed"
            sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}
          >
            <Toolbar>
              <Typography variant="h6" noWrap>
                Wine Classification Dashboard
              </Typography>
            </Toolbar>
          </AppBar>
          <Drawer
            variant="permanent"
            sx={{
              width: drawerWidth,
              flexShrink: 0,
              [`& .MuiDrawer-paper`]: {
                width: drawerWidth,
                boxSizing: "border-box",
              },
            }}
          >
            <Toolbar />
            <Box sx={{ overflow: "auto" }}>
              <List>
                {["Dashboard", "Logs", "Docs"].map((text) => (
                  <ListItem
                    button
                    key={text}
                    selected={selectedMenu === text}
                    onClick={() => setSelectedMenu(text)}
                  >
                    <ListItemText primary={text} />
                  </ListItem>
                ))}
              </List>
            </Box>
          </Drawer>
          <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
            <Toolbar />
            {selectedMenu === "Logs" ? (
              <Container maxWidth="lg">
                <Paper sx={{ p: 2 }}>
                  <Box
                    display="flex"
                    justifyContent="space-between"
                    alignItems="center"
                  >
                    <Typography variant="h6">Logs</Typography>
                    <Button
                      variant="outlined"
                      size="small"
                      onClick={() => setLogs("")}
                    >
                      Clear
                    </Button>
                  </Box>
                  <Box
                    sx={{
                      whiteSpace: "pre-wrap",
                      fontFamily: "monospace",
                      maxHeight: 500,
                      overflow: "auto",
                      backgroundColor: "#f9f9f9",
                      p: 2,
                    }}
                  >
                    {logs}
                    <div ref={logsEndRef} />
                  </Box>
                </Paper>
              </Container>
            ) : selectedMenu === "Docs" ? (
            <Container maxWidth="lg">
  <Paper sx={{ p: 3 }}>
   <Typography variant="h4" gutterBottom>
        📘 Interface Documentation
      </Typography>
      <Typography variant="body1" paragraph>
        This is the official documentation for using the <strong>Wine Analysis Web Interface</strong>. It explains how to configure models, run analyses, and interpret results within the GUI.
      </Typography>
      <Typography variant="body1" paragraph>
        🔗 For <strong>general project documentation</strong>, including installation instructions, backend structure, and data formats, please visit:
        <br />
        👉 <a href="https://pougetlab.github.io/wine_analysis/" target="_blank" rel="noopener noreferrer">pougetlab.github.io/wine_analysis</a>
      </Typography>

    <Accordion>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography><strong>Overview</strong></Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Typography>
            This tool is a GC-MS wine analysis application that allows you to load chromatographic data and analyze it using various machine learning models.
            <br />
            - For <strong>Bordeaux, Pinot Noir,</strong> and <strong>Press</strong> wines, the interface supports training and evaluation of <em>classification models</em>.
            <br />
            - For <strong>Champagne</strong> wines, the interface supports prediction of <em>sensory attributes</em> using <em>regression</em> models.
            <br />
            Additionally, you can:
            <ul>
              <li>Visualize 2D and 3D <strong>dimensionality reduction</strong> plots (PCA, UMAP, t-SNE).</li>
              <li>Generate and inspect <strong>confusion matrices</strong>.</li>
              <li>Compare performance across models and features.</li>
              <li>For Champagne, run taster-based evaluations and compare <strong>individual vs. group predictions</strong>.</li>
              <li>View system logs and model outputs.</li>
            </ul>        </Typography>
            <Typography paragraph>
              The interface is <strong>dynamic</strong>: depending on your selected wine family and task, the available options and visualizations will adapt automatically. For instance, classification tasks may display confusion matrices and dimensionality reduction plots, while regression tasks focus on attribute prediction and error metrics.
            </Typography>

      </AccordionDetails>
    </Accordion>

    <Accordion>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography><strong>Wine Families</strong></Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Typography>
          The application supports datasets from four main wine families:
          <ul>
            <li><strong>Bordeaux</strong>: Typically used for winery and year  classification</li>
            <li><strong>Pinot Noir</strong>: Used for winery-level origin classification but also supports origin, specific regions or even country </li>
            <li><strong>Press Wines</strong>: Samples labeled according to press type (A, B, or C), used for classifying the type of wine press employed.</li>
            <li><strong>Champagne</strong>: Samples evaluated by multiple tasters and annotated with multiple sensory attributes
            (e.g., fruity, citrus, toasted, honey, etc.). This dataset supports tasks such as modeling individual taster behavior,
            comparing tasters to group consensus or predicting sensory profiles from GC-MS chromatographic data
            </li>
          </ul>
          <p>Note: Dataset paths may need to be edited in the file <strong>config.yaml</strong> to match the location of your local data files.</p>
        </Typography>
      </AccordionDetails>
    </Accordion>

    <Accordion>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography><strong>Configuration Parameters</strong></Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Typography component="div">
          <ul>
            <li><strong>Wine Family</strong>: Choose the dataset</li>
            <li><strong>Model Type</strong>: Ridge, Random Forest, 1D CNN</li>
            <li><strong>Group Wines</strong>: Average ratings across tasters</li>
            <li><strong>Normalize</strong>: Standardize chromatogram intensity per wine</li>
            <li><strong>Subtask</strong>: Enables specific experimental modes like shuffled baseline or individual taster modeling</li>
          </ul>
        </Typography>
      </AccordionDetails>
    </Accordion>

    <Accordion>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography><strong>Supported Tasks</strong></Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Typography>
          <ul>
            <li><strong>Train Model</strong>: Trains a regression model per task settings</li>
            <li><strong>Evaluate Model</strong>: Cross-validation with MAE, RMSE, R² per attribute</li>
            <li><strong>Compare Tasters</strong>: Models each taster and compares with consensus model</li>
            <li><strong>Shuffle Baseline</strong>: Randomly shuffles y-values to assess chance performance</li>
            <li><strong>Consensus Modeling</strong>: Uses one-hot encoded tasters to capture individual bias</li>
          </ul>
        </Typography>
      </AccordionDetails>
    </Accordion>

    <Accordion>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography><strong>Outputs & Metrics</strong></Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Typography>
          Output depends on the selected task. Typical metrics include:
          <ul>
            <li><strong>R²</strong>: Coefficient of determination</li>
            <li><strong>MAE</strong>: Mean absolute error per attribute</li>
            <li><strong>RMSE</strong>: Root mean square error</li>
            <li><strong>Heatmaps</strong>: Taster-wise or attribute-wise prediction accuracy</li>
            <li><strong>Model Weights</strong>: Ridge coefficients or CNN filters</li>
          </ul>
        </Typography>
      </AccordionDetails>
    </Accordion>

    <Accordion>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography><strong>FAQs</strong></Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Typography>
          <ul>
            <li><strong>Why is R² negative?</strong> — Your model performs worse than a constant predictor</li>
            <li><strong>Can I upload my own data?</strong> — Currently not supported</li>
            <li><strong>What is Group Wines?</strong> — It averages ratings across tasters for each wine</li>
          </ul>
        </Typography>
      </AccordionDetails>
    </Accordion>
  </Paper>
</Container>

              ) : (
              <Container maxWidth="lg">
                <Paper sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Run Configuration
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={3}>
                      <FormControl fullWidth variant="outlined">
                        <InputLabel shrink>Wine Family</InputLabel>
                       <Select
                          value={wineFamily}
                          onChange={(e) => {
                            const selectedScript = e.target.value;
                            setWineFamily(selectedScript);

                            // Reset region if not pinot
                            if (selectedScript !== "pinot") {
                              setRegion(null);
                            }

                            // Set default dataset if available
                            if (defaultDatasetByScript[selectedScript]) {
                              setSelectedDatasets(defaultDatasetByScript[selectedScript]);
                            } else if (selectedScript === "bordeaux") {
                              setSelectedDatasets("bordeaux_oak");
                            } else if (selectedScript === "press") {
                              setSelectedDatasets("press_rep1");
                            } else if (selectedScript === "pinot") {
                              setSelectedDatasets("pinot_vineyard");
                            } else if (selectedScript === "champagne") {
                              setSelectedDatasets("heterocyc");
                            } else {
                              setSelectedDatasets(datasetOptions[selectedScript]);
                            }
                          }}
                          label="Wine Family"
                        >
                          <MenuItem value="bordeaux">Bordeaux</MenuItem>
                          <MenuItem value="pinot">Pinot Noir</MenuItem>
                          <MenuItem value="press">Press Wines</MenuItem>
                          <MenuItem value="champagne">Champagne</MenuItem>
                        </Select>
                      </FormControl>

                      {wineFamily === "pinot" && (
                        <Grid item xs={12} md={3}>
                          <Grid container spacing={2} sx={{ mt: 1 }}>
                            <FormControl fullWidth variant="outlined">
                              <InputLabel shrink>Region</InputLabel>
                              <Select
                                value={region}
                                onChange={(e) => setRegion(e.target.value)}
                                label="Region"
                              >
                                <MenuItem value="winery">Winery</MenuItem>
                                <MenuItem value="origin">Origin</MenuItem>
                                <MenuItem value="country">Country</MenuItem>
                                <MenuItem value="continent">Continent</MenuItem>
                                <MenuItem value="burgundy">
                                  N/S Burgundy
                                </MenuItem>
                              </Select>
                            </FormControl>
                          </Grid>
                        </Grid>
                      )}
                    </Grid>
                    <Grid item xs={12}>
                      <FormControl fullWidth variant="outlined">
                        <InputLabel id="task-label">Task</InputLabel>
                        <Select
                          labelId="task-label"
                          value={selectedTask}
                          label="Task"
                          onChange={(e) => setSelectedTask(e.target.value)}
                          displayEmpty
                        >
                          {taskOptions.map((task) => (
                            <MenuItem key={task} value={task}>
                              {task}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12}>
                      <FormControl fullWidth variant="outlined">
                        <InputLabel shrink>Datasets</InputLabel>
                        <Select
                          multiple
                          value={selectedDatasets}
                          onChange={(e) => setSelectedDatasets(e.target.value)}
                          renderValue={(selected) => selected.join(", ")}
                          label="Dataset"
                        >
                          {(wineFamily === "champagne" &&
                          selectedTask === "Predict Labels"
                            ? ["sensory_scores"]
                            : datasetOptions[wineFamily] || []
                          ).map((d) => (
                            <MenuItem key={d} value={d}>
                              {d}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                    {wineFamily === "champagne" &&
                      selectedTask === "Predict Labels" && (
                        <Grid item xs={12} md={3}>
                          <FormControl fullWidth variant="outlined">
                            <InputLabel shrink>Target</InputLabel>
                            <Select
                              multiple
                              value={labelTargets}
                              onChange={(e) => setLabelTargets(e.target.value)}
                              renderValue={(selected) => selected.join(", ")}
                              label="Target"
                            >
                              <MenuItem value="taster">Taster</MenuItem>
                              <MenuItem value="prod area">
                                Production Area
                              </MenuItem>
                              <MenuItem value="variety">Variety</MenuItem>
                              <MenuItem value="cave">Cave</MenuItem>
                              <MenuItem value="age">Age</MenuItem>
                            </Select>
                          </FormControl>
                        </Grid>
                      )}
                  {(wineFamily !== "champagne") || (wineFamily === "champagne" && selectedTask === "Predict Age" && doClassification) ? (
  // Classifier dropdown for:
  // - all non-Champagne wines, OR
  // - Champagne + Predict Age + doClassification = true
  <Grid item xs={12} md={3}>
    <FormControl fullWidth variant="outlined">
      <InputLabel shrink>Classifier</InputLabel>
      <Select
        value={classifier || ""}
        onChange={(e) => setClassifier(e.target.value)}
        label="Classifier"
      >
        <MenuItem value="DTC">Decision Tree</MenuItem>
        <MenuItem value="GNB">Gaussian Naive Bayes</MenuItem>
        <MenuItem value="KNN">K-Nearest Neighbors</MenuItem>
        <MenuItem value="LDA">Linear Discriminant Analysis</MenuItem>
        <MenuItem value="LR">Logistic Regression</MenuItem>
        <MenuItem value="PAC">Passive Aggressive</MenuItem>
        <MenuItem value="PER">Perceptron</MenuItem>
        <MenuItem value="RFC">Random Forest</MenuItem>
        <MenuItem value="RGC">Ridge Classifier</MenuItem>
        <MenuItem value="SGD">Stochastic Gradient Descent</MenuItem>
        <MenuItem value="SVM">Support Vector Machine</MenuItem>
      </Select>
    </FormControl>
  </Grid>
) : (
  // Regressor dropdown for:
  // - Champagne + other tasks (or doClassification = false)
  <Grid item xs={12} md={3}>
    <FormControl fullWidth variant="outlined">
      <InputLabel shrink>Regressor</InputLabel>
      <Select
        value={regressor || ""}
        onChange={(e) => setRegressor(e.target.value)}
        label="Regressor"
      >
        <MenuItem value="ridge">Ridge</MenuItem>
        <MenuItem value="lasso">Lasso</MenuItem>
        <MenuItem value="elasticnet">ElasticNet</MenuItem>
        <MenuItem value="rf">Random Forest</MenuItem>
        <MenuItem value="hgb">HistGradient Boosting</MenuItem>
        <MenuItem value="svr">Support Vector Regr.</MenuItem>
        <MenuItem value="knn">KNN</MenuItem>
        <MenuItem value="dt">Decision Tree</MenuItem>
        <MenuItem value="xgb">XGBoost</MenuItem>
      </Select>
    </FormControl>
  </Grid>
)}
{/*                     {wineFamily === "champagne" && */}
{/*                     (selectedTask === "Predict Age" || */}
{/*                       selectedTask === "Model Global" || */}
{/*                       selectedTask === "Model per Taster") ? ( */}
{/*                       <Grid item xs={12} md={3}> */}
{/*                         <FormControl fullWidth variant="outlined"> */}
{/*                           <InputLabel shrink>Regressor</InputLabel> */}
{/*                           <Select */}
{/*                             value={regressor || ""} */}
{/*                             onChange={(e) => setRegressor(e.target.value)} */}
{/*                             label="Regressor" */}
{/*                           > */}
{/*                             <MenuItem value="ridge">Ridge</MenuItem> */}
{/*                             <MenuItem value="lasso">Lasso</MenuItem> */}
{/*                             <MenuItem value="elasticnet">ElasticNet</MenuItem> */}
{/*                             <MenuItem value="rf">Random Forest</MenuItem> */}
{/*                             <MenuItem value="gbr">Gradient Boosting</MenuItem> */}
{/*                             <MenuItem value="hgb"> */}
{/*                               HistGradient Boosting */}
{/*                             </MenuItem> */}
{/*                             <MenuItem value="svr"> */}
{/*                               Support Vector Regr. */}
{/*                             </MenuItem> */}
{/*                             <MenuItem value="knn">KNN</MenuItem> */}
{/*                             <MenuItem value="dt">Decision Tree</MenuItem> */}
{/*                             <MenuItem value="xgb">XGBoost</MenuItem> */}
{/*                           </Select> */}
{/*                         </FormControl> */}
{/*                       </Grid> */}
{/*                     ) : ( */}
{/*                       <Grid item xs={12} md={3}> */}
{/*                         <FormControl fullWidth variant="outlined"> */}
{/*                           <InputLabel shrink>Classifier</InputLabel> */}
{/*                           <Select */}
{/*                             value={classifier || ""} */}
{/*                             onChange={(e) => setClassifier(e.target.value)} */}
{/*                             label="Classifier" */}
{/*                           > */}
{/*                             <MenuItem value="DTC">Decision Tree</MenuItem> */}
{/*                             <MenuItem value="GNB"> */}
{/*                               Gaussian Naive Bayes */}
{/*                             </MenuItem> */}
{/*                             <MenuItem value="KNN">K-Nearest Neighbors</MenuItem> */}
{/*                             <MenuItem value="LDA"> */}
{/*                               Linear Discriminant Analysis */}
{/*                             </MenuItem> */}
{/*                             <MenuItem value="LR">Logistic Regression</MenuItem> */}
{/*                             <MenuItem value="PAC">Passive Aggressive</MenuItem> */}
{/*                             <MenuItem value="PER">Perceptron</MenuItem> */}
{/*                             <MenuItem value="RFC">Random Forest</MenuItem> */}
{/*                             <MenuItem value="RGC">Ridge Classifier</MenuItem> */}
{/*                             <MenuItem value="SGD"> */}
{/*                               Stochastic Gradient Descent */}
{/*                             </MenuItem> */}
{/*                             <MenuItem value="SVM"> */}
{/*                               Support Vector Machine */}
{/*                             </MenuItem> */}
{/*                           </Select> */}
{/*                         </FormControl> */}
{/*                       </Grid> */}
{/*                     )} */}
                    {!(
                      wineFamily === "champagne" &&
                      selectedTask === "Predict Labels"
                    ) && (
                      <Grid item xs={12} md={3}>
                        <FormControl
                          fullWidth
                          sx={{ width: 120 }}
                          variant="outlined"
                        >
                          <InputLabel shrink>Feature Type</InputLabel>
                          <Select
                            value={featureType}
                            onChange={(e) => setFeatureType(e.target.value)}
                            label="Feature type"
                          >
                          <MenuItem value="tic">TIC</MenuItem>
                          <MenuItem value="tis" disabled={wineFamily === "champagne"}>TIS</MenuItem>
                          <MenuItem value="tic_tis" disabled={wineFamily === "champagne"}>TIC + TIS</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                    )}
                    <Grid item xs={12} sm={6} md={4}>
                      {wineFamily !== "champagne" && (
                      <FormControl fullWidth>
                        <InputLabel id="cv-type-label">CV</InputLabel>
                        <Select
                          labelId="cv-type-label"
                          value={cvType}
                          label="CV"
                          onChange={(e) => setCvType(e.target.value)}
                        >
                          <MenuItem value="LOOPC">LOOPC</MenuItem>
                          <MenuItem value="LOO" disabled={!(wineFamily === "pinot" || wineFamily === "bordeaux")}>LOO</MenuItem>
                          <MenuItem value="stratified" disabled={!(wineFamily === "pinot" || wineFamily === "bordeaux")}>Strat.</MenuItem>
                        </Select>
                      </FormControl>
                      )}
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <TextField
                        fullWidth
                        label="Repeats"
                        type="number"
                        value={numRepeats}
                        onChange={(e) => setNumRepeats(Number(e.target.value))}
                        sx={{ width: 80 }}
                      />
                    </Grid>
                    {/*               <Grid item xs={12} md={3}> */}
                    {/*                 <TextField fullWidth */}
                    {/*                 label="Chrom Cap" */}
                    {/*                 type="number" */}
                    {/*                 value={chromCap} onChange={(e) => */}
                    {/*                 setChromCap(Number(e.target.value))} */}
                    {/*                 sx={{ width: 90 }} */}
                    {/*                 /> */}
                    {/*               </Grid> */}
                    <Grid item xs={12} md={6}>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <TextField
                            sx={{ width: 100 }}
                            label="Min RT"
                            type="number"
                            value={rtRange[0]}
                            onChange={(e) => {
                              const newMin = Number(e.target.value);
                              if (newMin <= rtRange[1]) {
                                setRtRange([newMin, rtRange[1]]);
                              }
                            }}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <TextField
                            sx={{ width: 100 }}
                            label="Max RT"
                            type="number"
                            value={rtRange[1]}
                            onChange={(e) => {
                              const newMax = Number(e.target.value);
                              if (newMax >= rtRange[0]) {
                                setRtRange([rtRange[0], newMax]);
                              }
                            }}
                          />
                        </Grid>
                      </Grid>
                    </Grid>

                    <Grid container spacing={2} sx={{ mt: 1 }}>
                      <Grid item xs={12} md={2}>
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={normalize}
                              onChange={() => setNormalize(!normalize)}
                            />
                          }
                          label="Normalize"
                        />
                      </Grid>

                      {/*               {!(wineFamily === "champagne" && selectedTask === "Predict Labels") && ( */}
                      {wineFamily === "champagne" &&
                      selectedTask === "Model Global" ? (
                        <>
                          <Grid item xs={12} md={3}>
                      <Tooltip title="Apply dimensionality reduction to sensory attributes">
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={reduceDims}
                              onChange={() => setReduceDims(!reduceDims)}
                            />
                          }
                          label="Reduce Dims"
                        />
                      </Tooltip>
                    </Grid>

                    {reduceDims && (
                      <>
                        <Grid item xs={12} md={3}>
                          <FormControl fullWidth sx={{ width: 100 }}>
                            <InputLabel id="reduction-method-label">Method</InputLabel>
                            <Select
                              labelId="reduction-method-label"
                              value={reductionMethod}
                              label="Reduction Method"
                              onChange={(e) => setReductionMethod(e.target.value)}
                            >
                              <MenuItem value="pca">PCA</MenuItem>
                              <MenuItem value="umap">UMAP</MenuItem>
                              <MenuItem value="tsne">t-SNE</MenuItem>
                            </Select>
                          </FormControl>
                        </Grid>

                        <Grid item xs={12} md={3}>
                          <FormControl fullWidth sx={{ width: 70 }}>
                            <InputLabel id="output-dims-label">Dims</InputLabel>
                            <Select
                              labelId="output-dims-label"
                              value={reductionDims}
                              label="Output Dimensions"
                              onChange={(e) => setReductionDims(Number(e.target.value))}
                            >
                              {[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].map((dim) => (
                                <MenuItem key={dim} value={dim}>
                                  {dim}
                                </MenuItem>
                              ))}
                            </Select>
                          </FormControl>
                        </Grid>
                      </>
                    )}
                          <Grid item xs={12} md={3}>
                            <Tooltip title="Shows taster's prediction profiles for the last repeat">
                              <FormControlLabel
                                control={
                                  <Checkbox
                                    checked={showChampPredictedProfiles}
                                    onChange={() =>
                                      setShowChampPredictedProfiles(
                                        !showChampPredictedProfiles,
                                      )
                                    }
                                  />
                                }
                                label="Show Taster Profiles"
                              />
                            </Tooltip>
                          </Grid>
                          <Grid item xs={12} md={3}>
                            <Tooltip
                              title="Group (taster, wine) pairs by wine during cross-validation to avoid chromatograms from the same
                               wine to be used in both train and test splits. This predicts scores on unseen wines"
                            >
                              <FormControlLabel
                                control={
                                  <Checkbox
                                    checked={groupWines}
                                    onChange={() => setGroupWines(!groupWines)}
                                  />
                                }
                                label="Group Wines"
                              />
                            </Tooltip>
                          </Grid>
                          <Grid item xs={12} md={6}>
                            <FormControl fullWidth sx={{ minWidth: 200 }}>
                              <InputLabel id="taster-test-label" shrink={true}>
                                Tests & Subtasks
                              </InputLabel>
                              <Select
                                labelId="taster-test-label"
                                id="taster-test-select"
                                label="Tests & Subtasks"
                                value={tasterTests} // single string now
                                onChange={(e) => setTasterTests(e.target.value)}
                                displayEmpty
                                renderValue={(selected) => {
                                  if (!selected) {
                                    return (
                                      <span style={{ color: "#888" }}>
                                        Select a test
                                      </span>
                                    ); // Safe placeholder
                                  }
                                  return (
                                    testOptions.find(
                                      (opt) => opt.value === selected,
                                    )?.label || selected
                                  );
                                }}
                              >
                                <MenuItem value="">
                                  <em>None</em>
                                </MenuItem>
                                {testOptions.map((option) => (
                                  <MenuItem
                                    key={option.value}
                                    value={option.value}
                                    disabled={
                                       groupWines &&
                                       (option.value === "average" || option.value === "vsmean")
                                    }
                                    title={option.tooltip} // browser tooltip
                                  >
                                    <ListItemText
                                      primary={option.label}
                                      secondary={option.tooltip}
                                      secondaryTypographyProps={{
                                        style: {
                                          fontSize: "0.75rem",
                                          color: "#777",
                                        },
                                      }}
                                    />
                                  </MenuItem>
                                ))}
                              </Select>
                            </FormControl>
                          </Grid>
                          <Grid item xs={12} md={3}>
                            <Tooltip
                              title="Shows a plot of R² values"
                            >
                              <FormControlLabel
                                control={
                                  <Checkbox
                                    checked={plotR2}
                                    onChange={(e) => setPlotR2(e.target.checked)}
                                    disabled={tasterTests === "average"}
                                  />
                                }
                                label="Plot R²"
                              />
                            </Tooltip>
                          </Grid>


                          {/*     <Grid item xs={12} md={3}> */}
                          {/*       <Tooltip title="Adds a scaling layer (sensitivity) for each taster during learning"> */}
                          {/*       <FormControlLabel */}
                          {/*         control={<Checkbox checked={useChampTasterScaling} onChange={() => setUseChampTasterScaling(!useChampTasterScaling)} />} */}
                          {/*         label="Taster Scaling" */}
                          {/*       /> */}
                          {/*       </Tooltip> */}
                          {/*     </Grid> */}
                          {/*     <Grid item xs={12} md={3}> */}
                          {/*      <Tooltip title="Randomly reassigns sensory scores  of each sample to chromatograms"> */}
                          {/*       <FormControlLabel */}
                          {/*         control={<Checkbox checked={shuffleLabels} onChange={() => setShuffleLabels(!shuffleLabels)} />} */}
                          {/*         label="Shuffle Labels" */}
                          {/*       /> */}
                          {/*      </Tooltip> */}
                          {/*     </Grid> */}
                          {/*     <Grid item xs={12} md={3}> */}
                          {/*      <Tooltip title="Trains and evaluates a model on average sensory scores, excluding individual tasters identity"> */}
                          {/*       <FormControlLabel */}
                          {/*         control={<Checkbox checked={testAverageScores} onChange={() => setTestAverageScores(!testAverageScores)} />} */}
                          {/*         label="Average Scores" */}
                          {/*       /> */}
                          {/*      </Tooltip> */}
                          {/*     </Grid> */}
                          {/*     <Grid item xs={12} md={3}> */}
                          {/*      <Tooltip title="Compares each individual taster with the average of the rest"> */}
                          {/*       <FormControlLabel */}
                          {/*         control={<Checkbox checked={tasterVsMean} onChange={() => setTasterVsMean(!tasterVsMean)} />} */}
                          {/*         label="Taster vs Mean" */}
                          {/*       /> */}
                          {/*      </Tooltip> */}
                          {/*     </Grid> */}
                        </>
                      ) : (
                        <>
                          {/* Show Sync Time & Classify by Year only if not hidden */}
                          {!(
                            wineFamily === "champagne" &&
                            (selectedTask === "Predict Labels" ||
                              selectedTask === "Predict Age" ||
                              selectedTask === "Model Global")
                          ) && (
                            <>
                              <Grid item xs={12} md={2}>
                                <FormControlLabel
                                  control={
                                    <Checkbox
                                      checked={syncState}
                                      onChange={() => setSyncState(!syncState)}
                                    />
                                  }
                                  label="Sync Time"
                                />
                              </Grid>
                              <Grid item xs={12} md={2}>
                                {["bordeaux", "press"].includes(wineFamily) && (
                                <FormControlLabel
                                  control={
                                    <Checkbox
                                      checked={classByYear}
                                      onChange={() =>
                                        setClassByYear(!classByYear)
                                      }
                                    />
                                  }
                                  label="Classify by Year"
                                />
                                )}
                              </Grid>
                            </>
                          )}

                          {/* Always allow Confusion Matrix unless Predict Labels or Model Global */}
                         {wineFamily === "champagne" && selectedTask === "Model per Taster" ? (
                              <>
                                <Grid item xs={12} md={3}>
                                  <FormControlLabel
                                    control={
                                      <Checkbox
                                        checked={showGlobalFocusHeatmap}
                                        onChange={() =>
                                          setShowGlobalFocusHeatmap(!showGlobalFocusHeatmap)
                                        }
                                      />
                                    }
                                    label="Global Focus Heatmap"
                                  />
                                </Grid>
                                <Grid item xs={12} md={3}>
                                  <FormControlLabel
                                    control={
                                      <Checkbox
                                        checked={showTasterFocusHeatmap}
                                        onChange={() =>
                                          setShowTasterFocusHeatmap(!showTasterFocusHeatmap)
                                        }
                                      />
                                    }
                                    label="Taster Focus Heatmap"
                                  />
                                </Grid>
                              </>
                            ) : (
                              !(wineFamily === "champagne" &&
                                (selectedTask === "Predict Labels" ||
                                  selectedTask === "Model Global")) && (
                                <Grid item xs={12} md={2}>
                                  <FormControlLabel
                                    control={
                                      <Checkbox
                                        checked={showConfusionMatrix}
                                        onChange={() =>
                                          setShowConfusionMatrix(!showConfusionMatrix)
                                        }
                                      />
                                    }
                                    label="Show Confusion Matrix"
                                  />
                                </Grid>
                              )
                            )}
                        </>
                      )}
                      {isChampagneAgePrediction && (
                        <>
                          <FormControlLabel
                            control={
                              <Checkbox
                                checked={showPredPlot}
                                onChange={(e) =>
                                  setShowPredPlot(e.target.checked)
                                }
                              />
                            }
                            label="Plot True vs Predicted"
                          />
                          <FormControlLabel
                            control={
                              <Checkbox
                                checked={showAgeHist}
                                onChange={(e) =>
                                  setShowAgeHist(e.target.checked)
                                }
                              />
                            }
                            label="Show Age Distribution"
                          />
                          <FormControlLabel
                            control={
                              <Checkbox
                                checked={showChroms}
                                onChange={(e) =>
                                  setShowChroms(e.target.checked)
                                }
                              />
                            }
                            label="Show Chromatograms"
                          />
                          <FormControlLabel
                          control={
                            <Checkbox
                              checked={doClassification}
                              onChange={(e) => setDoClassification(e.target.checked)}
                            />
                          }
                          label="Do Classification Instead"
                        />
                        </>
                      )}
                    </Grid>
                  </Grid>
                  {!(
                    (wineFamily === "champagne" &&
                      selectedTask === "Predict Age") ||
                    selectedTask === "Model Global" ||
                    selectedTask === "Model per Taster"
                  ) && (
                    <Grid
                      container
                      spacing={2}
                      alignItems="center"
                      sx={{ mt: 2 }}
                    >
                      <Grid item xs={12} md={2}>
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={plotProjection}
                              onChange={(e) =>
                                setPlotProjection(e.target.checked)
                              }
                            />
                          }
                          label="Plot Projection"
                        />
                      </Grid>

                      {plotProjection && (
                        <>
                          {wineFamily === "bordeaux" && (
                            <>
                              <Grid item xs={12} md={2}>
                                <FormControl sx={{ width: 130, mr: 2 }}>
                                  <InputLabel>Projection Method</InputLabel>
                                  <Select
                                    value={projectionMethod}
                                    onChange={(e) =>
                                      setProjectionMethod(e.target.value)
                                    }
                                    label="Projection Method"
                                  >
                                    <MenuItem value="UMAP">UMAP</MenuItem>
                                    <MenuItem value="t-SNE">t-SNE</MenuItem>
                                    <MenuItem value="PCA">PCA</MenuItem>
                                  </Select>
                                </FormControl>
                              </Grid>

                              <Grid item xs={12} md={2}>
                                <FormControl sx={{ width: 150 }}>
                                  <InputLabel>Projection Source</InputLabel>
                                  <Select
                                    value={projectionSource}
                                    onChange={(e) =>
                                      setProjectionSource(e.target.value)
                                    }
                                    label="Projection Source"
                                  >
                                    {featureToProjectionOptions[
                                      featureType
                                    ].map((option) => (
                                      <MenuItem
                                        key={option.value}
                                        value={option.value}
                                      >
                                        {option.label}
                                      </MenuItem>
                                    ))}
                                  </Select>
                                </FormControl>
                              </Grid>

                              <Grid item xs={12} md={2}>
                                <TextField
                                  label="Dim."
                                  type="number"
                                  value={projectionDim}
                                  onChange={(e) => {
                                    const val = parseInt(e.target.value);
                                    if (val >= 2 && val <= 3)
                                      setProjectionDim(val);
                                  }}
                                  inputProps={{ min: 2, max: 3 }}
                                  fullWidth
                                  sx={{ width: 90 }}
                                />
                              </Grid>

                              <Grid item xs={12} md={3}>
                                {projectionMethod !== "PCA" && (
                                  <>
                                    {projectionMethod === "t-SNE" ? (
                                      <TextField
                                        label="Perplexity"
                                        type="number"
                                        value={perplexity}
                                        onChange={(e) =>
                                          setPerplexity(Number(e.target.value))
                                        }
                                        fullWidth
                                        sx={{ width: 90 }}
                                      />
                                    ) : (
                                      <TextField
                                        label="Neighbors"
                                        type="number"
                                        value={nNeighbors}
                                        onChange={(e) =>
                                          setNNeighbors(Number(e.target.value))
                                        }
                                        fullWidth
                                        sx={{ width: 90 }}
                                      />
                                    )}
                                  </>
                                )}
                              </Grid>

                              <Grid item xs={12} md={3}>
                                <TextField
                                  label="Rand. State"
                                  type="number"
                                  value={randomState}
                                  onChange={(e) =>
                                    setRandomState(Number(e.target.value))
                                  }
                                  fullWidth
                                  sx={{ width: 90 }}
                                />
                              </Grid>
                              <FormGroup row>
                                  <FormControlLabel
                                    control={
                                      <Checkbox checked={invertX} onChange={(e) => setInvertX(e.target.checked)} />
                                    }
                                    label="Invert X axis"
                                  />
                                  <FormControlLabel
                                    control={
                                      <Checkbox checked={invertY} onChange={(e) => setInvertY(e.target.checked)} />
                                    }
                                    label="Invert Y axis"
                                  />
                                </FormGroup>
                            </>
                          )}
                        </>
                      )}

                      {plotProjection && (
                        <>
                          {wineFamily === "press" && (
                            <>
                              <Grid item xs={12} md={2}>
                                <FormControl sx={{ width: 130, mr: 2 }}>
                                  <InputLabel>Projection Method</InputLabel>
                                  <Select
                                    value={projectionMethod}
                                    onChange={(e) =>
                                      setProjectionMethod(e.target.value)
                                    }
                                    label="Projection Method"
                                  >
                                    <MenuItem value="UMAP">UMAP</MenuItem>
                                    <MenuItem value="t-SNE">t-SNE</MenuItem>
                                    <MenuItem value="PCA">PCA</MenuItem>
                                  </Select>
                                </FormControl>
                              </Grid>

                              <Grid item xs={12} md={2}>
                                <FormControl sx={{ width: 150 }}>
                                  <InputLabel>Projection Source</InputLabel>
                                  <Select
                                    value={projectionSource}
                                    onChange={(e) =>
                                      setProjectionSource(e.target.value)
                                    }
                                    label="Projection Source"
                                  >
                                    {featureToProjectionOptions[
                                      featureType
                                    ].map((option) => (
                                      <MenuItem
                                        key={option.value}
                                        value={option.value}
                                      >
                                        {option.label}
                                      </MenuItem>
                                    ))}
                                  </Select>
                                </FormControl>
                              </Grid>

                              <Grid item xs={12} md={2}>
                                <TextField
                                  label="Dim."
                                  type="number"
                                  value={projectionDim}
                                  onChange={(e) => {
                                    const val = parseInt(e.target.value);
                                    if (val >= 2 && val <= 3)
                                      setProjectionDim(val);
                                  }}
                                  inputProps={{ min: 2, max: 3 }}
                                  fullWidth
                                  sx={{ width: 90 }}
                                />
                              </Grid>

                              <Grid item xs={12} md={3}>
                                {projectionMethod !== "PCA" && (
                                  <>
                                    {projectionMethod === "t-SNE" ? (
                                      <TextField
                                        label="Perplexity"
                                        type="number"
                                        value={perplexity}
                                        onChange={(e) =>
                                          setPerplexity(Number(e.target.value))
                                        }
                                        fullWidth
                                        sx={{ width: 90 }}
                                      />
                                    ) : (
                                      <TextField
                                        label="Neighbors"
                                        type="number"
                                        value={nNeighbors}
                                        onChange={(e) =>
                                          setNNeighbors(Number(e.target.value))
                                        }
                                        fullWidth
                                        sx={{ width: 90 }}
                                      />
                                    )}
                                  </>
                                )}
                              </Grid>

                              <Grid item xs={12} md={3}>
                                <TextField
                                  label="Rand. State"
                                  type="number"
                                  value={randomState}
                                  onChange={(e) =>
                                    setRandomState(Number(e.target.value))
                                  }
                                  fullWidth
                                  sx={{ width: 90 }}
                                />
                              </Grid>
                            </>
                          )}
                        </>
                      )}

                      {plotProjection && (
                        <>
                          {wineFamily === "pinot" && (
                            <>
                              <Grid item xs={12} md={2}>
                                <FormControl sx={{ width: 130, mr: 2 }}>
                                  <InputLabel>Projection Method</InputLabel>
                                  <Select
                                    value={projectionMethod}
                                    onChange={(e) =>
                                      setProjectionMethod(e.target.value)
                                    }
                                    label="Projection Method"
                                  >
                                    <MenuItem value="UMAP">UMAP</MenuItem>
                                    <MenuItem value="t-SNE">t-SNE</MenuItem>
                                    <MenuItem value="PCA">PCA</MenuItem>
                                  </Select>
                                </FormControl>
                              </Grid>

                              <Grid item xs={12} md={2}>
                                <FormControl sx={{ width: 150 }}>
                                  <InputLabel>Projection Source</InputLabel>
                                  <Select
                                    value={projectionSource}
                                    onChange={(e) =>
                                      setProjectionSource(e.target.value)
                                    }
                                    label="Projection Source"
                                  >
                                    {featureToProjectionOptions[
                                      featureType
                                    ].map((option) => (
                                      <MenuItem
                                        key={option.value}
                                        value={option.value}
                                      >
                                        {option.label}
                                      </MenuItem>
                                    ))}
                                  </Select>
                                </FormControl>
                              </Grid>

                              <Grid item xs={12} md={2}>
                                <TextField
                                  label="Dim."
                                  type="number"
                                  value={projectionDim}
                                  onChange={(e) => {
                                    const val = parseInt(e.target.value);
                                    if (val >= 2 && val <= 3)
                                      setProjectionDim(val);
                                  }}
                                  inputProps={{ min: 2, max: 3 }}
                                  fullWidth
                                  sx={{ width: 90 }}
                                />
                              </Grid>

                              <Grid item xs={12} md={3}>
                                {projectionMethod !== "PCA" && (
                                  <>
                                    {projectionMethod === "t-SNE" ? (
                                      <TextField
                                        label="Perplexity"
                                        type="number"
                                        value={perplexity}
                                        onChange={(e) =>
                                          setPerplexity(Number(e.target.value))
                                        }
                                        fullWidth
                                        sx={{ width: 90 }}
                                      />
                                    ) : (
                                      <TextField
                                        label="Neighbors"
                                        type="number"
                                        value={nNeighbors}
                                        onChange={(e) =>
                                          setNNeighbors(Number(e.target.value))
                                        }
                                        fullWidth
                                        sx={{ width: 90 }}
                                      />
                                    )}
                                  </>
                                )}
                              </Grid>

                              <Grid item xs={12} md={3}>
                                <TextField
                                  label="Rand. State"
                                  type="number"
                                  value={randomState}
                                  onChange={(e) =>
                                    setRandomState(Number(e.target.value))
                                  }
                                  fullWidth
                                  sx={{ width: 90 }}
                                />
                              </Grid>
                              <FormGroup row>
                                  <FormControlLabel
                                    control={
                                      <Checkbox checked={invertX} onChange={(e) => setInvertX(e.target.checked)} />
                                    }
                                    label="Invert X axis"
                                  />
                                  <FormControlLabel
                                    control={
                                      <Checkbox checked={invertY} onChange={(e) => setInvertY(e.target.checked)} />
                                    }
                                    label="Invert Y axis"
                                  />
                                </FormGroup>
                                <div>
                                <label>
                                  <input
                                    type="checkbox"
                                    checked={showSampleNames}
                                    onChange={(e) =>
                                      setShowSampleNames(e.target.checked)
                                    }
                                  />
                                  Show sample names
                                </label>
                              </div>
                            </>
                          )}
                        </>
                      )}

                      {plotProjection && (
                        <>
                          {wineFamily === "champagne" && (
                            <>
                              <Grid item xs={12} md={2}>
                                <FormControl sx={{ width: 130, mr: 2 }}>
                                  <InputLabel>Projection Method</InputLabel>
                                  <Select
                                    value={projectionMethod}
                                    onChange={(e) =>
                                      setProjectionMethod(e.target.value)
                                    }
                                    label="Projection Method"
                                  >
                                    <MenuItem value="UMAP">UMAP</MenuItem>
                                    <MenuItem value="t-SNE">t-SNE</MenuItem>
                                    <MenuItem value="PCA">PCA</MenuItem>
                                  </Select>
                                </FormControl>
                              </Grid>

                              <Grid item xs={12} md={2}>
                                <FormControl sx={{ width: 175 }}>
                                  <InputLabel>Projection Source</InputLabel>
                                  <Select
                                    value={projectionSource}
                                    onChange={(e) =>
                                      setProjectionSource(e.target.value)
                                    }
                                    label="Projection Source"
                                  >
                                    {(wineFamily === "champagne" &&
                                    selectedTask === "Predict Labels"
                                      ? champagnePredictLabelsProjectionOptions
                                      : featureToProjectionOptions[featureType]
                                    ).map((option) => (
                                      <MenuItem
                                        key={option.value}
                                        value={option.value}
                                      >
                                        {option.label}
                                      </MenuItem>
                                    ))}
                                  </Select>
                                </FormControl>
                              </Grid>

                              <Grid item xs={12} md={2}>
                                <TextField
                                  label="Dim."
                                  type="number"
                                  value={projectionDim}
                                  onChange={(e) => {
                                    const val = parseInt(e.target.value);
                                    if (val >= 2 && val <= 3)
                                      setProjectionDim(val);
                                  }}
                                  inputProps={{ min: 2, max: 3 }}
                                  fullWidth
                                  sx={{ width: 90 }}
                                />
                              </Grid>

                              <Grid item xs={12} md={3}>
                                {projectionMethod !== "PCA" && (
                                  <>
                                    {projectionMethod === "t-SNE" ? (
                                      <TextField
                                        label="Perplexity"
                                        type="number"
                                        value={perplexity}
                                        onChange={(e) =>
                                          setPerplexity(Number(e.target.value))
                                        }
                                        fullWidth
                                        sx={{ width: 90 }}
                                      />
                                    ) : (
                                      <TextField
                                        label="Neighbors"
                                        type="number"
                                        value={nNeighbors}
                                        onChange={(e) =>
                                          setNNeighbors(Number(e.target.value))
                                        }
                                        fullWidth
                                        sx={{ width: 90 }}
                                      />
                                    )}
                                  </>
                                )}
                              </Grid>

                              <Grid item xs={12} md={3}>
                                <TextField
                                  label="Rand. State"
                                  type="number"
                                  value={randomState}
                                  onChange={(e) =>
                                    setRandomState(Number(e.target.value))
                                  }
                                  fullWidth
                                  sx={{ width: 90 }}
                                />
                              </Grid>
                            </>
                          )}
                        </>
                      )}
                    </Grid>
                  )}
                  <Box sx={{ mt: 3 }}>
                    <Button
                      variant="contained"
                      onClick={run}
                      disabled={loading}
                    >
                      {loading ? "Running..." : "Run Script"}
                    </Button>
                  </Box>
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle1">Output</Typography>
                    <Paper
                      ref={outputRef}
                      variant="outlined"
                      sx={{
                        p: 2,
                        maxHeight: 500,
                        overflow: "auto",
                        backgroundColor: "#f9f9f9",
                      }}
                    >
                      <pre>{output}</pre>
                    </Paper>
                  </Box>
                </Paper>
              </Container>
            )}
          </Box>
        </Box>
      }
    </ThemeProvider>
  );
}

export default App;
