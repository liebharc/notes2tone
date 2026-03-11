### File Structure

```.
├── README.md
├── generated
│   ├── pred
│   │   ├── sample_00000.krn
│   │   ├── ...
│   └── gt
│       ├── sample_00000.krn
│       ├── ...
├── audiveris_output
│   ├── sample_00000.mxl
│   ├── ...
└── audiveris_output_scaled
    ├── sample_00000.mxl
    ├── ...
```

In the gt folder, you can find the ground truth files in krn format provided by the SMB dataset. In the pred folder, you can find the predicted files in krn format (the musicxml files from audiveris_output_scaled converted with xml2hum). The audiveris_output folder contains the original output from Audiveris in mxl format, while the audiveris_output_scaled folder contains the output from Audiveris where the images where scaled by 2 beforehand.

### Copy Audiveris XML files

Use `loadxml.py` to copy MusicXML files (`.xml` and `.mxl`) into generated folders:

- `audiveris_output` -> `generated/xml`
- `audiveris_output_scaled` -> `generated/xml_scaled`

Example:

```bash
python loadxml.py
```

Dry-run (no file writes):

```bash
python loadxml.py --dry-run
```
