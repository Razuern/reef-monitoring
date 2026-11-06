# Hyperspectral Imaging (HSI)

- Hyperspectral imaging collects and processes information from across the electromagnetic spectrum. The goal of hyperspectral imaging is to obtain the spectrum for each pixel in the image of a scene, with the purpose of finding objects, identifying materials, or detecting processes. There are three general types of spectral imagers. There are push broom scanners and the related whisk broom scanners (spatial scanning), which read images over time, band sequential scanners (spectral scanning), which acquire images of an area at different wavelengths, and snapshot hyperspectral imagers, which uses a staring array to generate an image in an instant.

- This technique of dividing images into bands can be extended beyond the visible. In hyperspectral imaging, the recorded spectra have fine wavelength resolution and cover a wide range of wavelengths. Hyperspectral imaging measures continuous spectral bands, as opposed to multiband imaging which measures spaced spectral bands.


## HSI in Coral Reef & *Zooxanthellae* Observation

In underwater coral reef monitoring, HSI typically operates across hundreds of narrow, contiguous bands (1–5 nm resolution) within the **400–720 nm** window (since water strongly attenuates near-infrared wavelengths above ~720 nm). Each pixel forms a continuous reflectance/fluorescence spectrum $(x, y, \lambda)$ that acts as a biochemical fingerprint of the coral holobiont.

### *Zooxanthellae* Pigment Fingerprinting
Unlike broad-band sensors, HSI directly resolves overlapping absorption troughs and reflectance/fluorescence peaks associated with photosynthetic and photoprotective pigments inside *Zooxanthellae* (dinoflagellates of the family *Symbiodiniaceae*):
- **Peridinin (Dinoflagellate-Specific Biomarker):** Strong absorption band across **470–550 nm** (peaking near **485–500 nm**), with a characteristic reflectance shoulder near **540–565 nm**. Because peridinin is unique to dinoflagellates, it allows HSI to distinguish *Zooxanthellae* inside healthy coral tissue from competing benthic macroalgae or cyanobacteria.
- **Chlorophyll-$a$ (Chl-$a$):** Primary photosynthetic pigment exhibiting sharp absorption troughs at **430–440 nm** (Soret band) and **665–675 nm**, alongside a natural solar-induced fluorescence emission peak at **683–685 nm**.
- **Chlorophyll-$c_2$ (Chl-$c_2$):** Accessory pigment with distinct absorption features near **450–460 nm** and **630–635 nm**.
- **Photoprotective Xanthophylls (Diadinoxanthin & Diatoxanthin):** Absorption shifts between **440–490 nm** reflect activation of the xanthophyll cycle under excess light and thermal stress.
- **Host Fluorescent Proteins (GFPs/RFPs):** Coral host pigments emit strongly between **480–520 nm** and **570–590 nm**, interacting with *Zooxanthellae* absorption to regulate internal light microclimates.

---

# Traditional Imaging Methods for Coral Health Observation

1. **Standard RGB Photography & Photogrammetry**
   - **Mechanism:** Captures three broad visible bands (Blue ~400–500 nm, Green ~500–600 nm, Red ~600–700 nm). Often paired with color reference charts (e.g., CoralWatch Health Chart) or deep learning classifiers (such as the ConvNeXt V2 binary classifier in `main.py`).
   - **Zooxanthellae Observation:** Infers symbiont concentration indirectly through macroscopic tissue brightness/paling (loss of brown/golden-brown coloration as *Zooxanthellae* are expelled, exposing the white calcium carbonate skeleton).
   - **Limitations:** Acts as a **lagging indicator**—by the time paling is visually or algorithmically separable in RGB, severe symbiont loss has already occurred. RGB also struggles to separate partial bleaching from sediment cover, lighting variations, or algal turf colonization.

2. **Multispectral Imaging (MSI)**
   - **Mechanism:** Captures 4 to ~15 discrete, spaced wavelength bands via satellites, UAVs, or filtered underwater cameras.
   - **Zooxanthellae Observation:** Improves large-scale mapping and separation of broad reef functional groups (live coral, bleached coral, macroalgae, sand), with better water-column attenuation correction than RGB.
   - **Limitations:** Spaced bands (>10–20 nm width) cannot resolve narrow pigment-specific features like the 632 nm Chl-$c_2$ trough or subtle red-edge shifts, limiting sub-lethal stress detection.

3. **Fluorescence Imaging & PAM Fluorometry**
   - **Mechanism:** Active excitation (using blue/UV LEDs or lasers) to measure chlorophyll fluorescence and Photosystem II (PSII) quantum yield ($F_v/F_m$).
   - **Zooxanthellae Observation:** Directly quantifies the photosynthetic efficiency and photoinhibition of *Zooxanthellae* prior to expulsion.
   - **Limitations:** Pulse-Amplitude Modulated (PAM) fluorometry is typically point-based rather than spatially imaged, while wide-field fluorescence imaging requires controlled excitation or nighttime deployment to avoid solar background interference.

---

# Predicting Future Coral Degradation via HSI Wavelengths

To forecast coral health degradation **before macroscopic bleaching occurs**, predictive models leverage early spectral shifts across key HSI wavelengths:

- **Xanthophyll & Peridinin-to-Chlorophyll Ratio Shifts (440–550 nm vs. 675 nm):**
  Under early thermal or irradiance stress, *Zooxanthellae* upregulate photoprotective xanthophylls (**440–490 nm**) while peridinin (**~490 nm**) and accessory pigments begin to degrade at different rates than Chl-$a$ (**~675 nm**). Tracking band ratios (e.g., $R_{490}/R_{675}$ or $R_{550}/R_{675}$) captures physiological stress days to weeks prior to symbiont expulsion.
- **Shallowing of the Red Chlorophyll Trough (665–675 nm):**
  A gradual increase in reflectance around **670–675 nm** indicates declining Chl-$a$ absorption efficiency and initial reduction in *Zooxanthellae* cell density before visible tissue whitening.
- **Red-Edge Inflection Shift (680–710 nm):**
  The steep reflectance transition between strong Chl-$a$ absorption (**675 nm**) and high near-infrared coral skeleton/tissue scattering (**700+ nm**) shifts toward shorter wavelengths (**blue-shift of the red edge**) during early pigment breakdown.
- **Peridinin Shoulder Flattening & Derivative Spectroscopy (540–650 nm):**
  Using first- and second-order spectral derivatives ($\frac{dR}{d\lambda}$, $\frac{d^2R}{d\lambda^2}$) between **500–680 nm** suppresses depth-dependent illumination artifacts and isolates subtle flattening of the peridinin reflectance shoulder (**~540–565 nm**) and the triple-peaked brown coral signature (**~575 nm, 600 nm, 650 nm**), providing robust early-warning features for predictive degradation models.

---

# Summary Comparison

| Criterion | Traditional RGB / MSI | Fluorescence / PAM | Hyperspectral Imaging (HSI) |
| :--- | :--- | :--- | :--- |
| **Spectral Resolution** | 3 broad bands (RGB) or 4–15 spaced bands (MSI) | Targeted excitation/emission bands | 100–300+ contiguous narrow bands (1–5 nm) |
| ***Zooxanthellae* Specificity** | **Low:** Indirect proxy via overall tissue brightness/color | **High:** Measures PSII photosynthetic efficiency ($F_v/F_m$) | **Very High:** Resolves Peridinin, Chl-$a$, Chl-$c_2$, and Xanthophylls |
| **Diagnostic Stage** | **Reactive:** Detects visible paling/bleaching after symbiont loss | **Proactive:** Detects early photoinhibition | **Predictive:** Detects sub-lethal pigment shifts & early density decline |
| **Key Predictive Wavelengths** | Broad R/G/B intensity ratios | 440–470 nm (Ex), 685 nm (Em) | **440–490 nm** (Xanthophylls), **490 & 545 nm** (Peridinin), **632 nm** (Chl-$c_2$), **675 nm** (Chl-$a$), **680–700 nm** (Red-edge) |
| **Deployment & Cost** | Low cost, simple deployment, standard CNNs | Moderate cost, point-based or active lighting required | High cost, requires radiometric/depth calibration & spectral-spatial models |
