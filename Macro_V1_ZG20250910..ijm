// CZI batch (2 channels: Green + Red, huge files) → Auto contrast (display-only)
// → SIFT MultiChannel (reg on C1) → save <base>_Aligned.tif (3D)
// → Split Channels (robust C1/C2 detect) → per-channel Max Z → Merge
// → save <base>_MAX_Aligned.tif (2D)
// Strong waits, per-file isolation, heap-based guards, optional physical RAM log.

// ===== Tunables =====
saturatedPct     = 0.35;    // Auto contrast (display only)
regChannel       = 1;       // Registration channel (1-based)
transformType    = "Rigid"; // Rigid / Similarity / Affine
maxWaitIters     = 2400;    // 2400*200ms = ~8 min open wait
postOpenWaitMs   = 3000;    // extra wait after open (ms)
afterSplitWait   = 500;     // wait after "Split Channels" (ms)
coolDownMs       = 500;     // pause between files (ms)
minHeapReserveMB = 2048;    // hard-stop if free heap < this after GC
useVirtual       = 1;       // 1 = open CZI as Virtual Stack (off-heap)
siftOnRAM        = 1;       // 1 = duplicate to RAM before SIFT (safer); 0 = run SIFT in-place
// ====================

// ---------- helpers (AVOID using 'i'/'j' used by outer loop) ----------
function waitForNewImage(beforeCount, maxIters) {
  it = 0;
  while (nImages() <= beforeCount && it < maxIters) { wait(200); it++; }
  return (nImages() > beforeCount) ? 1 : 0;
}
function latestImageTitle() { ts = getList("image.titles"); return ts[ts.length - 1]; }
function safeSetCalibration(pxW, pxH, pxD, unitStr) {
  if (pxW <= 0) pxW = 1.0; if (pxH <= 0) pxH = 1.0; if (pxD <= 0) pxD = 1.0;
  if (unitStr == "") unitStr = "micron";
  setVoxelSize(pxW, pxH, pxD, unitStr);
}
function closeAllImagesAndGC() {
  if (nImages() > 0) run("Close All");
  wait(250);
  run("Collect Garbage");
  wait(250);
}
function listDiff(before, after) {
  out = newArray();
  for (ii=0; ii<after.length; ii++) {
    seen = 0;
    for (jj=0; jj<before.length; jj++) if (after[ii] == before[jj]) { seen = 1; break; }
    if (seen == 0) out = Array.concat(out, after[ii]);
  }
  return out;
}
function b2mb(b) { return floor(parseFloat(b)/1048576); }
function heapStats() {
  maxB  = call("ij.IJ.maxMemory");      // JVM max (bytes)
  usedB = call("ij.IJ.currentMemory");  // JVM used (bytes)
  maxMB  = b2mb(maxB);
  usedMB = b2mb(usedB);
  freeMB = maxMB - usedMB;
  return newArray(maxMB, usedMB, freeMB);
}
// very rough op memory estimate (MB)
function estimateOpMB(W,H,Z,C,depthBits, extraFrames) {
  bytesPerPix = depthBits/8;
  core = W*H*maxOf(Z,1)*maxOf(C,1)*bytesPerPix;
  extra = W*H*maxOf(extraFrames,0)*bytesPerPix;
  return floor(1.2*(core + extra)/1048576);
}
// hard-stop if not enough heap headroom
function ensureHeapHeadroom(stage, needMB) {
  hs = heapStats(); // [max, used, free]
  print("Heap @ " + stage + " → max=" + hs[0] + "MB, used=" + hs[1] + "MB, free=" + hs[2] + "MB");
  if (hs[2] >= maxOf(needMB, minHeapReserveMB)) return 1;
  print("⚠️ Low heap headroom. GC…");
  run("Collect Garbage"); wait(500);
  hs = heapStats();
  print("After GC → max=" + hs[0] + "MB, used=" + hs[1] + "MB, free=" + hs[2] + "MB");
  if (hs[2] < maxOf(needMB, minHeapReserveMB)) {
    setBatchMode(false);
    run("Close All");
    run("Collect Garbage");
    exit("❌ Not enough JVM heap at '" + stage + "'. Need ~" + maxOf(needMB, minHeapReserveMB) + "MB free, have " + hs[2] + "MB. Increase Fiji -Xmx or set useVirtual=1.");
  }
  return 1;
}
// safe Z-projection using before/after diff; duplicate for 2D
function safeMaxProject(winTitle, tmpName) {
  if (!isOpen(winTitle)) return "";
  selectWindow(winTitle);
  getDimensions(w,h,c,z,t);
  safeSetCalibration(pxW, pxH, pxD, unitStr);

  if (z > 1) {
    before = getList("image.titles");
    run("Z Project...", "projection=[Max Intensity]");
    wait(250);
    after  = getList("image.titles");
    news   = listDiff(before, after);

    prj = "";
    for (kk=0; kk<news.length; kk++) {
      if (matches(news[kk], "^(MAX_|Max_).*$")) { prj = news[kk]; break; }
    }
    if (prj == "") {
      if (news.length > 0) prj = news[news.length-1];
      else prj = getTitle();
    }
  } else {
    run("Duplicate...", "title=" + tmpName);
    prj = tmpName;
  }

  if (!isOpen(prj)) return "";
  selectWindow(prj); safeSetCalibration(pxW, pxH, 1.0, unitStr);
  return prj;
}
// map split windows to C1/C2 via regex + color hints; robust fallback
function mapSplitWindows(newWins) {
  C1 = ""; C2 = "";
  // pass 1: "C1..." / "C2..."
  for (ll=0; ll<newWins.length; ll++) {
    t = newWins[ll];
    if (matches(t, "^C1.*")) C1 = t;
    else if (matches(t, "^C2.*")) C2 = t;
  }
  // pass 2: "(green)" / "(red)" hints
  if (C1=="" || C2=="") {
    for (mm=0; mm<newWins.length; mm++) {
      lt = toLowerCase(newWins[mm]);
      if (indexOf(lt, "(green)") >= 0 && C1=="") C1 = newWins[mm];
      else if (indexOf(lt, "(red)") >= 0 && C2=="") C2 = newWins[mm];
    }
  }
  // pass 3: fallback—first two windows
  if (C1=="" || C2=="") {
    if (newWins.length >= 2) {
      if (C1=="") C1 = newWins[0];
      if (C2=="") C2 = newWins[1];
    }
  }
  if (C1 == C2 && newWins.length >= 2) { C1 = newWins[0]; C2 = newWins[1]; }
  return newArray(C1, C2);
}
// (optional) best-effort log of physical RAM via JavaScript (prints to Log)
function logPhysicalRAM() {
  code = ""
    + "importPackage(java.lang);\n"
    + "var os = java.lang.management.ManagementFactory.getOperatingSystemMXBean();\n"
    + "try {\n"
    + "  var clazz = java.lang.Class.forName('com.sun.management.OperatingSystemMXBean');\n"
    + "  if (clazz.isInstance(os)) {\n"
    + "    var freeB = os.getFreePhysicalMemorySize();\n"
    + "    var totB  = os.getTotalPhysicalMemorySize();\n"
    + "    IJ.log('Physical RAM (free/total MB): ' + Math.floor(freeB/1048576) + ' / ' + Math.floor(totB/1048576));\n"
    + "  } else { IJ.log('Physical RAM: N/A'); }\n"
    + "} catch (e) { IJ.log('Physical RAM check failed: ' + e); }\n";
  run("Script...", "language=JavaScript code="+code);
}

// ---- RAM copy purge (kept exactly as requested) ----
function purgeRAMCopy(alignedTitleMaybe) {
  if (isOpen("RAM_Copy") && "RAM_Copy" != alignedTitleMaybe) {
    close("RAM_Copy");
    wait(100);
  }
  run("Collect Garbage");
  wait(200);
}

// -------------------------
setBatchMode(true);

// Choose folder
dir = getDirectory("Choose a folder with CZI files");
if (dir == "") { setBatchMode(false); exit("No folder selected."); }
list = getFileList(dir);

// Collect CZI files
cziList = newArray();
for (fileScanIdx=0; fileScanIdx<list.length; fileScanIdx++)
  if (endsWith(toLowerCase(list[fileScanIdx]), ".czi"))
    cziList = Array.concat(cziList, list[fileScanIdx]);

nFiles = cziList.length;
if (nFiles == 0) { setBatchMode(false); exit("No CZI files found."); }

print("\\Clear"); print("Found " + nFiles + " CZI file(s).");
logPhysicalRAM();

for (fileIdx=0; fileIdx<nFiles; fileIdx++) {
  // Proactively purge any stale RAM copy from previous file
  purgeRAMCopy(""); 

  closeAllImagesAndGC();
  ensureHeapHeadroom("start of file " + (fileIdx+1), 0);

  filename = cziList[fileIdx];
  base     = replace(replace(filename, ".czi",""), ".CZI","");
  print("\\n=== [" + (fileIdx+1) + "/" + nFiles + "] " + filename + " ===");

  // --- Open CZI (no autoscale) ---
  beforeCount = nImages();
  openArgs = "open=[" + dir + filename + "] color_mode=Default view=Hyperstack stack_order=XYCZT";
  if (useVirtual) openArgs = openArgs + " use_virtual_stack";
  run("Bio-Formats Importer", openArgs);
  opened = waitForNewImage(beforeCount, maxWaitIters);
  if (opened == 0) { print("❌ Timeout opening: " + filename); continue; }
  wait(postOpenWaitMs);

  // Select source window
  srcTitle = latestImageTitle();
  if (indexOf(srcTitle, base) < 0) {
    tsTmp = getList("image.titles");
    for (tt=0; tt<tsTmp.length; tt++) if (indexOf(tsTmp[tt], base) >= 0) { srcTitle = tsTmp[tt]; break; }
  }
  selectWindow(srcTitle);

  // calibration + dims
  getVoxelSize(pxW, pxH, pxD, unitStr);
  getDimensions(W,H,C,Z,T);
  bd = bitDepth(); // 8/16/32
  print("Dims: " + W + "x" + H + "  C=" + C + "  Z=" + Z + "  T=" + T);
  print("Voxel: " + pxW + " x " + pxH + " x " + pxD + " " + unitStr);
  if (C != 2) print("⚠️ Expected 2 channels, got C=" + C + " (continuing).");

  ensureHeapHeadroom("after open", estimateOpMB(W,H,1,1,bd,3));

  // Auto contrast (display only)
  for (cc=1; cc<=C; cc++) { Stack.setChannel(cc); run("Enhance Contrast", "saturated=" + saturatedPct); }

  // --- SIFT MultiChannel registration ---
  purgeRAMCopy(""); 
  if (siftOnRAM) {
    run("Duplicate...", "title=RAM_Copy duplicate");
    regTitle = "RAM_Copy";
  } else {
    regTitle = srcTitle;
  }
  selectWindow(regTitle);

  titlesBefore = getList("image.titles");
  run("Linear Stack Alignment with SIFT MultiChannel",
      "registration_channel=" + regChannel + " initial_gaussian_blur=1.60 steps_per_scale_octave=3 "
      + "minimum_image_size=64 maximum_image_size=1024 feature_descriptor_size=4 "
      + "feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 "
      + "maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=" + transformType + " interpolate");
  wait(200);
  titlesAfter = getList("image.titles");
  newAfter = listDiff(titlesBefore, titlesAfter);

  // detect aligned window (prefer one containing 'Aligned')
  alignedTitle = "";
  for (kk=0; kk<titlesAfter.length; kk++) if (indexOf(titlesAfter[kk], "Aligned") >= 0) { alignedTitle = titlesAfter[kk]; break; }
  if (alignedTitle == "") {
    if (newAfter.length > 0) alignedTitle = newAfter[newAfter.length-1];
    else alignedTitle = getTitle();
  }
  if (!isOpen(alignedTitle)) alignedTitle = regTitle;
  selectWindow(alignedTitle);

  // Save aligned 3D (keep calibration)
  safeSetCalibration(pxW, pxH, pxD, unitStr);
  alignedOut = dir + base + "_Aligned.tif";
  saveAs("Tiff", alignedOut);
  print("Saved aligned 3D: " + alignedOut);

  // purge RAM copy after saving aligned
  purgeRAMCopy(alignedTitle);

  ensureHeapHeadroom("after aligned save", estimateOpMB(W,H,1,1,bd,3));

  // --- Split Channels ---
  splitBefore = getList("image.titles");
  run("Split Channels");
  wait(afterSplitWait);
  splitAfter  = getList("image.titles");
  newWins     = listDiff(splitBefore, splitAfter);
  print("Split produced " + newWins.length + " windows:");
  for (ww=0; ww<newWins.length; ww++) print("  - " + newWins[ww]);

  pair = mapSplitWindows(newWins);
  C1title = pair[0]; C2title = pair[1];

  if (C1title=="" || C2title=="") {
    print("❌ Could not find split channel windows (C1/C2). Skipping.");
    for (ww=0; ww<newWins.length; ww++) if (isOpen(newWins[ww])) close(newWins[ww]);
    if (isOpen(alignedTitle)) close(alignedTitle);
    if (isOpen(srcTitle))     close(srcTitle);
    closeAllImagesAndGC(); wait(coolDownMs);
    continue;
  }

  // --- Per-channel Max Z ---
  prj1 = safeMaxProject(C1title, "TempC1");
  if (prj1=="") {
    print("❌ Max projection failed for " + C1title + ". Skipping.");
    for (ww=0; ww<newWins.length; ww++) if (isOpen(newWins[ww])) close(newWins[ww]);
    if (isOpen(alignedTitle)) close(alignedTitle);
    if (isOpen(srcTitle))     close(srcTitle);
    closeAllImagesAndGC(); wait(coolDownMs);
    continue;
  }
  rename("MAX_C1");
  if (isOpen(C1title)) close(C1title);

  prj2 = safeMaxProject(C2title, "TempC2");
  if (prj2=="") {
    print("❌ Max projection failed for " + C2title + ". Skipping.");
    for (ww=0; ww<newWins.length; ww++) if (isOpen(newWins[ww])) close(newWins[ww]);
    if (isOpen("MAX_C1")) close("MAX_C1");
    if (isOpen(alignedTitle)) close(alignedTitle);
    if (isOpen(srcTitle))     close(srcTitle);
    closeAllImagesAndGC(); wait(coolDownMs);
    continue;
  }
  rename("MAX_C2");
  if (isOpen(C2title)) close(C2title);

  ensureHeapHeadroom("after projections", estimateOpMB(W,H,1,2,bd,2));

  // --- Merge back to 2D multi-channel (C1=Green, C2=Red) ---
  run("Merge Channels...", "c1=MAX_C1 c2=MAX_C2 create");
  mergedTitle = getTitle();
  safeSetCalibration(pxW, pxH, 1.0, unitStr);
  run("Make Composite"); Stack.setChannel(1); Stack.setChannel(2);

  maxOut = dir + base + "_MAX_Aligned.tif";
  saveAs("Tiff", maxOut);
  print("Saved MaxIP 2D: " + maxOut);

  // purge RAM copy again right after final save
  purgeRAMCopy(alignedTitle);

  // cleanup this file
  if (isOpen(mergedTitle)) close(mergedTitle);
  if (isOpen("MAX_C1")) close("MAX_C1");
  if (isOpen("MAX_C2")) close("MAX_C2");
  if (isOpen(alignedTitle)) close(alignedTitle);
  if (isOpen(srcTitle))     close(srcTitle);

  closeAllImagesAndGC();
  wait(coolDownMs);
}

// Final cleanup and EXIT Fiji
setBatchMode(false);
run("Close All");
run("Collect Garbage");
print("\\n==============================");
print("✅ All files processed: " + nFiles);
print("==============================\\n");
run("Beep");
// Optional toast (comment out if unattended):
// showMessage("Done", "All files processed: " + nFiles);
exit();
