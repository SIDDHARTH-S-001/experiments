import adsk.core, adsk.fusion, adsk.cam, traceback

# Debugger: only used if you're attaching from VS Code
try:
    import debugpy
    debugpy.listen(('localhost', 9000))
    debugpy.wait_for_client()  # Optional: pause here until VS Code attaches
except:
    pass  # ignore if debugpy isn't available

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = app.activeProduct

        # Get root component of active design
        rootComp = design.rootComponent

        # Create a new sketch on the XY plane
        sketches = rootComp.sketches
        xyPlane = rootComp.xYConstructionPlane
        sketch = sketches.add(xyPlane)

        # Draw a 10x10 rectangle
        sketch.sketchCurves.sketchLines.addTwoPointRectangle(
            adsk.core.Point3D.create(0, 0, 0),
            adsk.core.Point3D.create(10, 10, 0)
        )

        # Create a profile and extrude it to 5mm
        prof = sketch.profiles.item(0)
        extrudes = rootComp.features.extrudeFeatures
        extInput = extrudes.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)

        distance = adsk.core.ValueInput.createByReal(5)
        extInput.setDistanceExtent(False, distance)
        extrudes.add(extInput)

    except Exception as e:
        if ui:
            ui.messageBox(f'Failed: {str(e)}\n{traceback.format_exc()}')
