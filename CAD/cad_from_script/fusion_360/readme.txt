FUSION 360 SCRIPT DEBUGGING SETUP (USING VSCODE)

REQUIREMENTS:
- Fusion 360 installed
- VS Code installed with the Python extension
- Python installed on your system (used by VS Code, not Fusion)
- `debugpy` is included in VS Code's Python extension

---------------------------------------
STEP 1: LOCATE FUSION PYTHON INTERPRETER
---------------------------------------

1. Create a temporary Fusion script with the following:
    import sys
    print(sys.executable)

2. Run the script via Fusion → Tools → Scripts and Add-Ins.

3. Note the printed path. It will look like:
   C:\Users\<you>\AppData\Local\Autodesk\webdeploy\production\<hash>\Python\python.exe

4. This is your PYTHON_EXE_PATH.

---------------------------------------
STEP 2: SET UP pre-run.py
---------------------------------------

1. Create a folder:
   C:\FusionDebug\pre_run_debug

2. Inside that folder, create a file named:
   pre-run.py

3. Paste the full content of pre-run.py you already have.

4. At the top of pre-run.py, set the following:

   VSCODE_PATH = 'C:\\Path\\To\\Code.exe'
   PYTHON_EXE_PATH = 'C:\\Path\\To\\Fusion\\Python\\python.exe'

   Replace both paths with actual values:
   - VS Code path is usually: C:\Users\<you>\AppData\Local\Programs\Microsoft VS Code\Code.exe
   - Python path is what you got in Step 1.

5. Save the file.

---------------------------------------
STEP 3: RUN pre-run.py IN FUSION
---------------------------------------

1. Open Fusion 360
2. Go to Tools → Scripts and Add-Ins
3. Click "Add"
4. Browse to C:\FusionDebug\pre_run_debug
5. Select pre-run.py
6. Click "Run"

This adds `debugpy` to Fusion's internal Python path.

---------------------------------------
STEP 4: CREATE YOUR ACTUAL SCRIPT
---------------------------------------

1. Create a folder for your working script, e.g.:
   D:\MyFusionScripts\box_creator

2. Inside that folder, create a file named:
   fusion_debug_runner.py

3. Paste the FusionDebugSetup class in that file.

4. In the same folder, create a file named:
   main.py

5. Paste the following content:

   import adsk.core, adsk.fusion, adsk.cam, traceback

   # Run pre-run.py dynamically
   try:
       from fusion_debug_runner import FusionDebugSetup
       debug_setup = FusionDebugSetup(r'C:\FusionDebug\pre_run_debug\pre-run.py')
       debug_setup.run()
   except Exception as debug_setup_error:
       print(f"[Warning] Failed to execute FusionDebugSetup: {debug_setup_error}")

   try:
       import debugpy
       debugpy.listen(('localhost', 9000))
       debugpy.wait_for_client()
   except:
       pass

   def run(context):
       ui = None
       try:
           app = adsk.core.Application.get()
           ui = app.userInterface
           design = app.activeProduct
           rootComp = design.rootComponent

           sketches = rootComp.sketches
           xyPlane = rootComp.xYConstructionPlane
           sketch = sketches.add(xyPlane)

           sketch.sketchCurves.sketchLines.addTwoPointRectangle(
               adsk.core.Point3D.create(0, 0, 0),
               adsk.core.Point3D.create(10, 10, 0)
           )

           prof = sketch.profiles.item(0)
           extrudes = rootComp.features.extrudeFeatures
           extInput = extrudes.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
           distance = adsk.core.ValueInput.createByReal(5)
           extInput.setDistanceExtent(False, distance)
           extrudes.add(extInput)

       except Exception as e:
           if ui:
               ui.messageBox(f'Failed: {str(e)}\n{traceback.format_exc()}')

6. Save both files.

---------------------------------------
STEP 5: SET UP launch.json FOR VSCODE
---------------------------------------

1. Inside D:\MyFusionScripts\box_creator, create a folder:
   .vscode

2. Inside .vscode, create a file named:
   launch.json

3. Paste the following:

   {
       "version": "0.2.0",
       "configurations": [
           {
               "name": "Python: Attach",
               "type": "python",
               "request": "attach",
               "connect": {
                   "host": "localhost",
                   "port": 9000
               },
               "pathMappings": [
                   {
                       "localRoot": "${workspaceFolder}",
                       "remoteRoot": "${workspaceFolder}"
                   }
               ]
           }
       ]
   }

4. Save the file.

---------------------------------------
STEP 6: RUN THE SCRIPT AND ATTACH DEBUGGER
---------------------------------------

1. In Fusion 360:
   - Go to Tools → Scripts and Add-Ins
   - Click Add
   - Browse to D:\MyFusionScripts\box_creator
   - Select main.py
   - Click Run

2. In VS Code:
   - Open folder: D:\MyFusionScripts\box_creator
   - Press F5 or click "Run and Debug" → Select "Python: Attach"

You are now debugging Fusion 360's Python script from VS Code.

Breakpoints, variable inspection, and stepping should now work.

---------------------------------------
DONE
---------------------------------------
