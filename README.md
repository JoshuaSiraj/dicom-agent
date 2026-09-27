# dicom-agent

A Google ADK agent that studies one local DICOM case by navigating the [med-datactrl](../med-datactrl) harness.

The model decides which series to inspect. The harness decides how a series is opened and how much comes back. The agent does not crawl folders itself.

## Shape

One `LlmAgent` (`dicom_agent/agent.py`). Its tools are built only from `TOOL_DEFINITIONS` in `med-datactrl/src/datactrl/tools.py`. Each ADK tool copies that definition's name, description, and parameters, and runs `datactrl.tools.dispatch`.

| Tool | Role |
| --- | --- |
| `discover` | Open a case folder and list series. Call this first. |
| `describe` | Modality, label, path, links, and fetch views. No pixels. |
| `fetch` | One bounded view. `preview` plus `slice_index` for an image series. `metadata` for an RTSTRUCT. |
| `dicom_groups` | Pair series. `query="CT,RTSTRUCT"` links a CT to its structure set. |
| `dicom_tree` | Reference forest of roots and children. |
| `links` | Structural links for one series or the whole case. |
| `assert_link` | Record a relationship the files did not already state. Session only. |

Resource ids look like `dicom:<SeriesInstanceUID>`. Later calls take `root` or the `case_id` returned by `discover`. `med-datactrl` keeps one `Case` per root.

## How a turn goes

1. `discover` the folder the user gave.
2. `describe` a series before fetching it.
3. `fetch` a preview to look at an image, or metadata to read an RTSTRUCT.
4. `dicom_groups` to pair an image series with its RTSTRUCT.
5. Report the series ids and what the tools returned.

## Later: PyRadiomics and predictive models

Not on the agent yet. Two tools would join the same way as the others: an entry in `TOOL_DEFINITIONS` and a handler behind `dispatch`.

- `radiomics` takes a linked image series, RTSTRUCT, and ROI name, and returns a small feature dict. The volume and mask stay inside the handler.
- `predict` scores a saved model on that dict.

Start with shape and first-order features. Published research signatures only, scored on a contoured CT target. The agent never sees the volume.

### Candidates

Only signatures whose numeric weights are published. Each scores a contoured CT target. They are research signatures, not clinical decision models.

| Model | Endpoint | Weights |
| --- | --- | --- |
| `lung_sbrt_local_control` | 1-year local control after lung SBRT | Logistic formula in the open-access paper |
| `nsclc_os_risk` | Overall survival, NSCLC | Ten Cox coefficients, open-access |
| `nsclc_io_response` | Immunotherapy response, advanced NSCLC | Seven-feature logistic score, open-access |
| `pdac_os` | Overall survival, resectable pancreatic cancer | Two-feature Cox signature, open-access |
| `aerts_2014` | Overall survival, NSCLC and head and neck | Four Cox coefficients from the external-validation table |

**Luo et al., 2022.** Planning CT, tumor ROI, 129 tumors (89 train, 40 validation). Radiomics-only validation AUC 0.70. A higher score meant a higher chance of local control at one year. The feature names match PyRadiomics:

```text
-27.645
+ 14.393 * wavelet-LLL_glszm_SmallAreaEmphasis
+ 8.075  * wavelet-LHH_glcm_JointAverage
- 3.386  * wavelet-LHH_ngtdm_Complexity
+ 9.196  * squareroot_glcm_DifferenceEntropy
```

Store that formula in `models/lung_sbrt_local_control.json`. No estimator file is required. [Frontiers in Oncology](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2021.819047/full).

**Le et al., 2021.** Pretreatment CT, tumor ROI, PyRadiomics names. A higher risk score meant shorter overall survival. Patients were split at the median score. [Cancers](https://doi.org/10.3390/cancers13143616):

```text
- 0.9936 * original_shape_Elongation
+ 0.005230 * original_gldm_DependenceVariance
- 0.1693 * wavelet-LHL_firstorder_Skewness
+ 4.764e-6 * wavelet-LHH_gldm_LargeDependenceHighGrayLevelEmphasis
- 0.1489 * wavelet-LHH_firstorder_Mean
- 6.460e-6 * wavelet-LLH_glcm_ClusterShade
+ 1.074e-4 * wavelet-LLH_firstorder_Maximum
+ 8.738e-9 * wavelet-HLH_firstorder_Energy
+ 0.3567 * wavelet-HLH_firstorder_Skewness
+ 2.988 * wavelet-HLL_glszm_GrayLevelNonUniformityNormalized
```

**Hou et al., 2025.** Contrast-enhanced CT, largest lung lesion, 144 patients (115 train, 29 test). Radiomics-model test AUC about 0.85. The score below is the radiomics part. The paper then mixes it with a clinical score. [Frontiers in Oncology](https://pmc.ncbi.nlm.nih.gov/articles/PMC12757218/):

```text
0.375
+ 0.075783 * original_firstorder_Maximum
+ 0.009822 * original_gldm_SmallDependenceLowGrayLevelEmphasis
- 0.042976 * original_glrlm_LongRunHighGrayLevelEmphasis
- 0.007042 * original_glszm_GrayLevelNonUniformity
- 0.033593 * original_glszm_SizeZoneNonUniformity
+ 0.046212 * original_glszm_SmallAreaLowGrayLevelEmphasis
+ 0.036860 * original_ngtdm_Coarseness
```

**Khalvati et al., 2019.** Preoperative CT, tumor ROI, two institutions (30 train, 68 validation). Validation hazard ratio 1.56 and 1.35 for the two readers. [Scientific Reports](https://doi.org/10.1038/s41598-019-41728-7):

```text
exp(
  0.44 * original_glcm_SumEntropy
  + 0.011 * squareroot_glcm_ClusterTendency
)
```

A 24-coefficient logistic score for hepatocellular carcinoma response to TACE is also published in full, in Table 2 of [Frontiers in Oncology, 2022](https://www.frontiersin.org/articles/10.3389/fonc.2022.853254/full). It is the same kind of model, with too many terms to repeat here.

**Aerts et al., 2014.** Pretreatment CT and GTV. Concordance about 0.65 on lung and 0.69 on head and neck. The weights below are the ones published for external validation ([Leijenaar et al., 2015](https://doi.org/10.3109/0284186X.2015.1061214)):

```text
+ 2.42e-11 * Energy
- 5.38e-3  * Compactness
- 1.47e-4  * GLRLM gray-level nonuniformity
+ 9.39e-6  * wavelet-HLH GLRLM gray-level nonuniformity
```

Those four features were defined in Matlab in 2014. PyRadiomics renamed Compactness to a different shape feature and defines Energy differently, so the weights apply only when the features are computed with the 2014 definitions. [Nature Communications](https://doi.org/10.1038/ncomms5006).

## Run

Python 3.12. `med-datactrl` does not support 3.14.

```bash
pixi install
# dicom_agent/.env
# GOOGLE_API_KEY=...
pixi run adk-web    # from this directory; open the dicom_agent app
pixi run adk-run
```
