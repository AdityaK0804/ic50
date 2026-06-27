const fs = require('fs');
const http = require('https');

function download(url, dest, cb) {
  const file = fs.createWriteStream(dest);
  http.get(url, function(response) {
    response.pipe(file);
    file.on('finish', function() {
      file.close(cb);
    });
  }).on('error', function(err) {
    fs.unlink(dest, () => {});
    if (cb) cb(err.message);
  });
}

const jsUrl = "https://unpkg.com/@rdkit/rdkit@2025.3.4-1.0.0/dist/RDKit_minimal.js";
const wasmUrl = "https://unpkg.com/@rdkit/rdkit@2025.3.4-1.0.0/dist/RDKit_minimal.wasm";

console.log("Downloading RDKit_minimal.js...");
download(jsUrl, "backend/RDKit_minimal.js", function(err) {
  if (err) {
    console.error("Failed to download JS:", err);
    process.exit(1);
  }
  console.log("Downloading RDKit_minimal.wasm...");
  download(wasmUrl, "backend/RDKit_minimal.wasm", function(err) {
    if (err) {
      console.error("Failed to download WASM:", err);
      process.exit(1);
    }
    runTest();
  });
});

function runTest() {
  console.log("Loading RDKit WASM...");
  const initRDKitModule = require('./RDKit_minimal.js');
  initRDKitModule().then(function(RDKit) {
    console.log("RDKit loaded successfully!");
    const mol = RDKit.get_mol("CC(=O)NC1=CC=C(C=C1)O");
    console.log("Molecule created.");
    
    // Print all properties on the molecule prototype
    const proto = Object.getPrototypeOf(mol);
    const methods = Object.getOwnPropertyNames(proto);
    console.log("\nAvailable JSMol methods:");
    console.log(methods.filter(m => m.includes("fp") || m.includes("fingerprint") || m.includes("morgan")).join(", "));
    
    // Let's test get_morgan_fp
    try {
      console.log("\nTesting get_morgan_fp()...");
      const fp = mol.get_morgan_fp();
      console.log("get_morgan_fp() result length:", fp.length);
      console.log("get_morgan_fp() result sample:", fp.substring(0, 50));
    } catch(e) {
      console.log("get_morgan_fp() failed:", e.message);
    }

    try {
      console.log("\nTesting get_morgan_fp with different JSON options...");
      
      const formats = [
        { radius: 2, len: 1024 },
        { radius: 2, nBits: 1024 },
        { radius: 2, numBits: 1024 },
        { radius: 2, n_bits: 1024 },
        { radius: 2, length: 1024 }
      ];
      
      for (const fmt of formats) {
        const fp = mol.get_morgan_fp(JSON.stringify(fmt));
        console.log(`Format ${JSON.stringify(fmt)} -> length: ${fp.length}`);
      }
    } catch(e) {
      console.log("get_morgan_fp failed:", e.message);
    }

    try {
      console.log("\nTesting get_morgan_fp_as_binary_text(2, 1024)...");
      const fp = mol.get_morgan_fp_as_binary_text(2, 1024);
      console.log("get_morgan_fp_as_binary_text(2, 1024) length:", fp.length);
      console.log("get_morgan_fp_as_binary_text(2, 1024) sample:", fp.substring(0, 50));
    } catch(e) {
      console.log("get_morgan_fp_as_binary_text(2, 1024) failed:", e.message);
    }

    process.exit(0);
  }).catch(function(err) {
    console.error("Initialization error:", err);
    process.exit(1);
  });
}
