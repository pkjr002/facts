def var_col_order(dataIP,columnId):
    try:
        slicedData = dataIP[columnId]
        #print("SlicedData" , slicedData)
        return slicedData
    except:
        #print("An exception occurred")
        return None
