import adsk.core, adsk.fusion, adsk.cam, traceback

# --- Import and run FusionDebugSetup ---
try:
    from fusionDebugSetup import FusionDebugSetup

    pre_run_script_path = r'C:\Users\SIDDHARTH\AppData\Roaming\Autodesk\Autodesk Fusion 360\API\Python\vscode\pre-run.py'
    debug_setup = FusionDebugSetup(pre_run_script_path)
    debug_setup.run()
except Exception as debug_setup_error:
    print(f"[Warning] Failed to execute FusionDebugSetup: {debug_setup_error}")

# --- Attach Debugger ---
try:
    import debugpy
    debugpy.listen(('localhost', 9000))
    debugpy.wait_for_client()  # Optional: pause until VS Code attaches
except:
    pass

# Global list to keep event handler references.
handlers = []

# ===============================================
# COMMAND DIALOG FOR CUBE CREATION WITH OPTIONS
# ===============================================

class CubeCreatorCommandCreatedHandler(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        try:
            cmd = args.command
            inputs = cmd.commandInputs
            
            # --- Dropdown for units ---
            dropdownUnits = inputs.addDropDownCommandInput('unitsInput', 'Units', adsk.core.DropDownStyles.TextListDropDownStyle)
            dropdownUnits.listItems.add('mm', True)    # default selected
            dropdownUnits.listItems.add('cm', False)
            dropdownUnits.listItems.add('inch', False)
            dropdownUnits.listItems.add('m', False)
            
            # --- Text boxes for dimensions (L, W, H) ---
            inputs.addStringValueInput('lengthInput', 'Length', '10')
            inputs.addStringValueInput('widthInput', 'Width', '10')
            inputs.addStringValueInput('heightInput', 'Height', '10')
            
            # --- Dropdown for feature type (None, Fillet, Chamfer) ---
            dropdownFeature = inputs.addDropDownCommandInput('featureType', 'Feature Type', adsk.core.DropDownStyles.TextListDropDownStyle)
            dropdownFeature.listItems.add('None', True)     # default is None
            dropdownFeature.listItems.add('Fillet', False)
            dropdownFeature.listItems.add('Chamfer', False)
            
            # --- Checkboxes for selection ---
            inputs.addBoolValueInput('edgeOption', 'Apply to Edges', True, '', True)
            inputs.addBoolValueInput('vertexOption', 'Apply to Vertices', False, '', False)
            
            # --- Text box for the feature value (radius/distance) ---
            inputs.addStringValueInput('featureValue', 'Feature Value', '1')
            
            # Add the execute handler.
            onExecute = CubeCreatorCommandExecuteHandler()
            cmd.execute.add(onExecute)
            handlers.append(onExecute)
        except Exception as e:
            app = adsk.core.Application.get()
            ui  = app.userInterface
            ui.messageBox('Failed in command created handler:\n{}'.format(traceback.format_exc()))

class CubeCreatorCommandExecuteHandler(adsk.core.CommandEventHandler):
    def notify(self, args):
        try:
            # Retrieve inputs.
            inputs = args.command.commandInputs
            unitItem = inputs.itemById('unitsInput').selectedItem.name
            length_val = float(inputs.itemById('lengthInput').value)
            width_val  = float(inputs.itemById('widthInput').value)
            height_val = float(inputs.itemById('heightInput').value)
            featureType = inputs.itemById('featureType').selectedItem.name
            edgeOption = inputs.itemById('edgeOption').value
            vertexOption = inputs.itemById('vertexOption').value
            feature_val = float(inputs.itemById('featureValue').value)
            
            # Conversion factors to convert user-entered values to Fusion's internal units.
            # (Assuming Fusion's design uses centimeters as the base unit.)
            conversion_factors = {'mm': 0.1, 'cm': 1.0, 'inch': 2.54, 'm': 100.0}
            factor = conversion_factors.get(unitItem, 1.0)
            length_converted = length_val * factor
            width_converted  = width_val  * factor
            height_converted = height_val * factor
            feature_converted = feature_val * factor
            
            # Get Fusion application and design info.
            app = adsk.core.Application.get()
            design = app.activeProduct
            rootComp = design.rootComponent

            # Create a new sketch on the XY construction plane.
            sketches = rootComp.sketches
            xyPlane = rootComp.xYConstructionPlane
            sketch = sketches.add(xyPlane)
            
            # Draw a rectangle for the base of the cube.
            pt1 = adsk.core.Point3D.create(0, 0, 0)
            pt2 = adsk.core.Point3D.create(length_converted, width_converted, 0)
            sketch.sketchCurves.sketchLines.addTwoPointRectangle(pt1, pt2)
            
            # Extrude the rectangle to create a cube.
            prof = sketch.profiles.item(0)
            extrudes = rootComp.features.extrudeFeatures
            extInput = extrudes.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
            heightInput = adsk.core.ValueInput.createByReal(height_converted)
            extInput.setDistanceExtent(False, heightInput)
            extFeature = extrudes.add(extInput)
            cubeBody = extFeature.bodies.item(0)
            
            # Apply fillet or chamfer, if chosen and if at least one checkbox is true.
            if featureType != 'None' and (edgeOption or vertexOption):
                if featureType == 'Fillet':
                    filletFeats = rootComp.features.filletFeatures
                    filletInput = filletFeats.createInput()
                    # For simplicity, if either option is checked, apply fillet to all edges.
                    edgesCollection = adsk.core.ObjectCollection.create()
                    for edge in cubeBody.edges:
                        edgesCollection.add(edge)
                    filletInput.addConstantRadiusEdgeSet(edgesCollection, adsk.core.ValueInput.createByReal(feature_converted), True)
                    filletFeats.add(filletInput)
                elif featureType == 'Chamfer':
                    chamferFeats = rootComp.features.chamferFeatures
                    # Create an ObjectCollection of all edges.
                    edgesCollection = adsk.core.ObjectCollection.create()
                    for edge in cubeBody.edges:
                        edgesCollection.add(edge)
                    # Use a simple equal-distance chamfer for demonstration.
                    chamferInput = chamferFeats.createInput2()
                    chamferInput.setToEqualDistance(edgesCollection, adsk.core.ValueInput.createByReal(feature_converted))
                    chamferFeats.add(chamferInput)
        except Exception as e:
            app = adsk.core.Application.get()
            ui  = app.userInterface
            ui.messageBox('Failed in command execute handler:\n{}'.format(traceback.format_exc()))

# ===============================================
# ENTRY POINT: CREATE AND EXECUTE THE COMMAND
# ===============================================
def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface
        
        # Create a command definition. (If one with the same ID already exists, use it.)
        cmdDef = ui.commandDefinitions.itemById('CubeCreatorCommand')
        if not cmdDef:
            cmdDef = ui.commandDefinitions.addButtonDefinition('CubeCreatorCommand', 'Cube Creator', 'Creates a cube with fillet/chamfer options')
        
        # Add the command created event handler.
        onCommandCreated = CubeCreatorCommandCreatedHandler()
        cmdDef.commandCreated.add(onCommandCreated)
        handlers.append(onCommandCreated)
        
        # Execute the command.
        cmdDef.execute()
        
        # Prevent Fusion 360 from terminating the script.
        adsk.autoTerminate(False)
    except Exception as e:
        if ui:
            ui.messageBox('Failed in run:\n{}'.format(traceback.format_exc()))
