function subscriber_callback(msg)
    -- This is the subscriber callback function
    simAddStatusbarMessage('subscriber receiver following Float32: '..msg.data)
end

function getTransformStamped(objHandle,name,relTo,relToName)
    -- This function retrieves the stamped transform for a specific object
    t=simGetSystemTime()
    p=simGetObjectPosition(objHandle,relTo)
    o=simGetObjectQuaternion(objHandle,relTo)
    return {
        header={
            stamp=t,
            frame_id=relToName
        },
        child_frame_id=name,
        transform={
            translation={x=p[1],y=p[2],z=p[3]},
            rotation={x=o[1],y=o[2],z=o[3],w=o[4]}
        }
    }
end

-- Functions to convert table to string
-- Produces a compact, uncluttered representation of a table. Mutual recursion is employed
-- http://lua-users.org/wiki/TableUtils
function table.val_to_str ( v )
  if "string" == type( v ) then
    v = string.gsub( v, "\n", "\\n" )
    if string.match( string.gsub(v,"[^'\"]",""), '^"+$' ) then
      return "'" .. v .. "'"
    end
    return '"' .. string.gsub(v,'"', '\\"' ) .. '"'
  else
    return "table" == type( v ) and table.tostring( v ) or
      tostring( v )
  end
end

function table.key_to_str ( k )
  if "string" == type( k ) and string.match( k, "^[_%a][_%a%d]*$" ) then
    return k
  else
    return "[" .. table.val_to_str( k ) .. "]"
  end
end

function table.tostring( tbl )
  local result, done = {}, {}
  for k, v in ipairs( tbl ) do
    table.insert( result, table.val_to_str( v ) )
    done[ k ] = true
  end
  for k, v in pairs( tbl ) do
    if not done[ k ] then
      table.insert( result,
        table.key_to_str( k ) .. "=" .. table.val_to_str( v ) )
    end
  end
  return "{" .. table.concat( result, "," ) .. "}"
end

if (sim_call_type==sim_childscriptcall_initialization) then
    -- The child script initialization
    objectHandle=simGetObjectAssociatedWithScript(sim_handle_self)
    objectName=simGetObjectName(objectHandle)
    -- Check if the required RosInterface is there:
    moduleName=0
    index=0
    rosInterfacePresent=false
    while moduleName do
        moduleName=simGetModuleName(index)
        if (moduleName=='RosInterface') then
            rosInterfacePresent=true
        end
        index=index+1
    end

    -- Prepare the float32 publisher and subscriber (we subscribe to the topic we advertise):
    if rosInterfacePresent then
        publisher=simExtRosInterface_advertise('/simulationTime','std_msgs/Float32')
        force_publisher=simExtRosInterface_advertise('/nico_feet_forces','std_msgs/String')
        subscriber=simExtRosInterface_subscribe('/simulationTime','std_msgs/Float32','subscriber_callback')
    end

    -- Get the graph handle
    if (simGetScriptExecutionCount()==0) then
        graphHandle=simGetObjectHandle("right_foot_avg_graph")
    end
end

if (sim_call_type==sim_childscriptcall_actuation) then
    -- Send an updated simulation time message, and send the transform of the object attached to this script:
    if rosInterfacePresent then
        simExtRosInterface_publish(publisher,{data=simGetSimulationTime()})

        -- Read the forces acting on the left leg
        l1_handle=simGetObjectHandle('left_sensor_1')
        l1_result, l1_forceVector, l1_torqueVector=simReadForceSensor(l1_handle)
        l2_handle=simGetObjectHandle('left_sensor_2')
        l2_result, l2_forceVector, l2_torqueVector=simReadForceSensor(l2_handle)
        l3_handle=simGetObjectHandle('left_sensor_3')
        l3_result, l3_forceVector, l3_torqueVector=simReadForceSensor(l3_handle)
        l4_handle=simGetObjectHandle('left_sensor_4')
        l4_result, l4_forceVector, l4_torqueVector=simReadForceSensor(l4_handle)

        -- Read the forces acting on the right leg
        r1_handle=simGetObjectHandle('right_sensor_1')
        r1_result, r1_forceVector, r1_torqueVector=simReadForceSensor(r1_handle)
        r2_handle=simGetObjectHandle('right_sensor_2')
        r2_result, r2_forceVector, r2_torqueVector=simReadForceSensor(r2_handle)
        r3_handle=simGetObjectHandle('right_sensor_3')
        r3_result, r3_forceVector, r3_torqueVector=simReadForceSensor(r3_handle)
        r4_handle=simGetObjectHandle('right_sensor_4')
        r4_result, r4_forceVector, r4_torqueVector=simReadForceSensor(r4_handle)

        -- Concatenate all the force arrays into a single string
        force_str = table.val_to_str(l1_forceVector)..','..
                    table.val_to_str(l2_forceVector)..','..
                    table.val_to_str(l3_forceVector)..','..
                    table.val_to_str(l4_forceVector)..','..
                    table.val_to_str(r1_forceVector)..','..
                    table.val_to_str(r2_forceVector)..','..
                    table.val_to_str(r3_forceVector)..','..
                    table.val_to_str(r4_forceVector)

        if l1_forceVector and l2_forceVector and l3_forceVector and l4_forceVector and r1_forceVector and r2_forceVector and r3_forceVector and r4_forceVector  then
            simAddStatusbarMessage('Force: '..force_str)
            simAddStatusbarMessage('Force2: '..l1_forceVector[1])
            simExtRosInterface_publish(force_publisher,{data=force_str})

            -- Compute the average of the left z-forces and the average of the right z-forces
            -- This is used for plotting
            right_z_avg = (r1_forceVector[3] + r2_forceVector[3] + r3_forceVector[3] + r4_forceVector[3])/4.0
            right_heel_z_avg = (r3_forceVector[3] + r4_forceVector[3])/2.0
            right_toe_z_avg = (r1_forceVector[3] + r2_forceVector[3])/2.0
            left_z_avg = (l1_forceVector[3] + l2_forceVector[3] + l3_forceVector[3] + l4_forceVector[3])/4.0
            left_heel_z_avg = (l3_forceVector[3] + l4_forceVector[3])/2.0
            left_toe_z_avg = (l1_forceVector[3] + l2_forceVector[3])/2.0

            -- Set the average values in the appropriate graph
            simSetGraphUserData(graphHandle, "r_heel_avg", right_heel_z_avg)
            simSetGraphUserData(graphHandle, "r_toe_avg", right_toe_z_avg)
            simSetGraphUserData(graphHandle, "r_avg", right_z_avg)
            simAddStatusbarMessage('Avg right heel (z): '..right_heel_z_avg)
        end
        simExtRosInterface_sendTransform(getTransformStamped(objectHandle,objectName,-1,'world'))
        -- To send several transforms at once, use simExtRosInterface_sendTransforms instead
    end
end

if (sim_call_type==sim_childscriptcall_cleanup) then
    -- Following not really needed in a simulation script (i.e. automatically shut down at simulation end):
    if rosInterfacePresent then
        simExtRosInterface_shutdownPublisher(publisher)
        simExtRosInterface_shutdownSubscriber(subscriber)
    end
end
