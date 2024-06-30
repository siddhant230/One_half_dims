### Dependencies for 1 and 2 :
```
!pip install -q diffusers
!pip install -U peft
!pip install -q accelerate
```
### SYSTEM 1 : setup text2image (done)
- simple fastapi based backend, input is text

### SYSTEM 2 : setup image_inpainting (done)
- simple fastapi based backend, input is text, init image and mask

To be done
- metric evaluations for VQI
- projection and gemini based evaluations for VQI
- UI planning
- Upgrade currend triposr gradio to use the entire setup wrt UI
- integration with Gradio in VQI tab

### SYSTEM 3 : setup triposr, this will serve as main backend and UI for everything (input is image)
    - text2image
    - img_inpaint
    - gemini based VQI
    - standard metrics
  - 
    ```
    %cd TripoSR
    update gradio_app with new_UI file
    !pip install --upgrade setuptools
    !pip install -r -q requirements.txt
    !python gradio_app.py --share
    ```

