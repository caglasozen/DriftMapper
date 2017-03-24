# DriftMapper

Maps and analyse drift in a dataset

# Usage
* Compile the codebase and generate a .jar artifact
* Call the .jar file with the following commands depending on your use case:
```sh
$ java -jar <path to .jar> arg1 arg2 arg3 arg4 arg5 ...
```
```
Comand line arguments:
analyse      subsetLength1,subsetLength2,... StartIndex1,EndIndex1,StartIndex2,EndIndex2 output_folder file1 file2 ...
stream       subsetLength1,subsetLength2,... windowSize1,windowSize2,...                 output_folder file1 file2 ...
stream_cont  subsetLength1,subsetLength2,... windowSize1,windowSize2,...                 output_folder file1 file2 ...
stream_chunk subsetLength1,subsetLength2,... groupAttIndex,groupSize1,groupsSize2,...    output_folder file1 file2 ...
moving_chunk subsetLength1,subsetLength2,... groupAttIndex,groupSize1,groupsSize2,...    output_folder file1 file2 ...
```
* See main.java file for examples

# Experiment types
## Analyse (analyse)
* Analyses the drift between 2 windows.
* Is able to output detailed analysis of the drift including the drift in the lielihood and posterior before averaging.

## Stream (stream)
* Comapares the drift between multiple fixed window sizes back to back
* eg. Compares drift between 1-10 and 11-20 then the drift between 11-20 and 21-30

## Stream Continuous (stream_cont)
* Compares the drift between 2 moving windows with fixed sizes
* eg. Compares drift between 1-10 and 11-20 then the drift between 2-11 and 12-21

## Chunk (stream_chunk)
* Groups the data into different chunks based on a given attribute
* Groups chunks of back-to-back instances with the same value of the given attribute to group on
* Given a group/time-window size, n, compares the first n chunk to the second n chunks and so on
* eg. Compares Chunks 1-2 and 3-4 then compares chunks 3-4 and 5-6

## Moving Chunk (moving_chunk)
* Same as experiment type Chunk
* Difference: given a group/time-window size, n, compares 2 moving chunk windows that moves by 1 chunk at a time
* eg. Compares Chunks 1-2 and 3-4 then compares chunks 2-3 and 4-5
