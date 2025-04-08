import adsk.core, adsk.fusion, adsk.cam, traceback

# --- Import and run FusionDebugSetup ---
try:
    from fusionDebugSetup import FusionDebugSetup

    pre_run_script_path = r'C:\Users\SIDDHARTH\AppData\Roaming\Autodesk\Autodesk Fusion 360\API\Python\vscode\pre-run.py'  # Update if different
    debug_setup = FusionDebugSetup(pre_run_script_path)
    debug_setup.run()
except Exception as debug_setup_error:
    print(f"[Warning] Failed to execute FusionDebugSetup: {debug_setup_error}")

# --- Attach Debugger ---
try:
    import debugpy
    debugpy.listen(('localhost', 9000))
    debugpy.wait_for_client()
except:
    pass

# --- Fusion 360 Model Creation ---
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

        # Creates a rectangle of 10x10
        sketch.sketchCurves.sketchLines.addTwoPointRectangle(
            adsk.core.Point3D.create(0, 0, 0),
            adsk.core.Point3D.create(10, 10, 0)
        )

        prof = sketch.profiles.item(0)
        extrudes = rootComp.features.extrudeFeatures
        extInput = extrudes.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        distance = adsk.core.ValueInput.createByReal(10) # extrude by 5 mm
        extInput.setDistanceExtent(False, distance)
        extrudes.add(extInput)

    except Exception as e:
        if ui:
            ui.messageBox(f'Failed: {str(e)}\n{traceback.format_exc()}')
