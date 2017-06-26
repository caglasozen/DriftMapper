package run;

import org.apache.commons.lang3.ArrayUtils;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.io.*;
import java.util.ArrayList;

/**
 * Created by loongkuan on 16/12/2016.
 */
public class main {
    /**
     * analyse      subsetLength1,subsetLength2,... StartIndex1,EndIndex1,StartIndex2,EndIndex2 output_folder file1 file2 ...
     * stream       subsetLength1,subsetLength2,... windowSize1,windowSize2,...                 output_folder file1 file2 ...
     * stream_cont  subsetLength1,subsetLength2,... windowSize1,windowSize2,...                 output_folder file1 file2 ...
     * stream_chunk subsetLength1,subsetLength2,... groupAttIndex,groupSize1,groupsSize2,...    output_folder file1 file2 ...
     * moving_chunk subsetLength1,subsetLength2,... groupAttIndex,groupSize1,groupsSize2,...    output_folder file1 file2 ...
     */
    public static void main(String[] argv) {
        // Example arguments
        argv = new String[]{"analyse", "1,2,3", "1,1991977,1991978,3983954", "./data_out/train_seed/20130505_20131129", "./datasets/train_seed/20130505.arff", "./datasets/train_seed/20131129.arff"};
        //argv = new String[]{"analyse", "1",  "800001,1100000,1100001,1400000", "./data_out/SITS_2006_NDVI_C/SITS1M_fold1_TEST", "./datasets/SITS_2006_NDVI_C/SITS1M_fold1_TEST.arff"};
        //argv = new String[]{"analyse", "1",  "3100001,3300000,3300001,3500000", "./data_out/SITS_2006_NDVI_C/SITS1M_fold1_TEST", "./datasets/SITS_2006_NDVI_C/SITS1M_fold1_TEST.arff"};
        //argv = new String[]{"analyse", "1,2,3,5",  "1,17376,17377,45312", "./data_out/elecNormNewClean", "./datasets/elecNormNewClean.arff"};
        //argv = new String[]{"analyse", "1,2,3,7",  "1,17376,17377,539383", "./data_out/airlines", "./datasets/airlines.arff"};
        /*
        argv = new String[]{"stream_chunk", "1,10",  "-1", "./data_out/train_seed",
                "./datasets/train_seed/20130910.arff",
                "./datasets/train_seed/20130926.arff",
                "./datasets/train_seed/20131012.arff",
                "./datasets/train_seed/20131028.arff",
                "./datasets/train_seed/20131113.arff",
                "./datasets/train_seed/20131129.arff"};
                */
        //argv = new String[]{"moving_chunk", "1,2,7",  "0,7,30", "./data_out/elecNormNew", "./datasets/elecNormNew.arff"};
        //argv = new String[]{"moving_chunk", "1,2,5",  "0,7,30", "./data_out/elecNormNewClean", "./datasets/elecNormNewClean.arff"};
        //argv = new String[]{"moving_chunk", "1,2",  "0,7,30", "./data_out/elecNormNewReduced", "./datasets/elecNormNewReduced.arff"};
        //argv = new String[]{"moving_chunk", "1",  "0,1", "./data_out/SITS_2006_NDVI_C/SITS1M_fold1_TEST", "./datasets/SITS_2006_NDVI_C/SITS1M_fold1_TEST.arff"};
        //argv = new String[]{"moving_chunk", "1,2,6",  "4,1,7", "./data_out/airlines", "./datasets/airlines.arff"};

        // Obtain folder too put outputs
        String[] folders = argv[3].split("/");
        String outputFolder = createFilePath(ArrayUtils.add(folders, argv[0]));

        // Get data from given .arff files
        String[] dataFiles = ArrayUtils.subarray(argv, 4, argv.length);
        Instances[] allInstances = new Instances[dataFiles.length];
        for (int i = 0; i < dataFiles.length; i++) {
            allInstances[i] = loadDataSet(dataFiles[i]);
            if (allInstances[i].classIndex() < 0) {
                allInstances[i].setClassIndex(allInstances[i].numAttributes() - 1);
            }
        }

        // Obtain Subset Length
        String[] subsetLengthsString = argv[1].split(",");
        int[] subsetLengths = new int[subsetLengthsString.length];
        for (int i = 0; i < subsetLengthsString.length; i++) {
            subsetLengths[i] = Integer.parseInt(subsetLengthsString[i]);
        }

        Instances instances;
        switch (argv[0]){
            //TODO: Different names for different windows
            case "analyse":
                folders = ArrayUtils.add(folders, argv[0]);
                folders = ArrayUtils.add(folders, argv[2].replaceAll(",", "_"));
                outputFolder = createFilePath(folders);
                String[] splitArgs = argv[2].split(",");
                instances = mergeInstances(allInstances);
                for (int i = 0; i < instances.numAttributes(); i++) {
                    if (instances.attribute(i).isDate()) {
                        instances.deleteAttributeAt(i);
                    }
                }
                instances = discretizeDataSet(instances);
                int[] splitIndices = new int[splitArgs.length];
                for (int i = 0; i < splitIndices.length; i++) {
                    double arg = Double.parseDouble(splitArgs[i]);
                    if (arg / (int)Math.floor(arg) == 1) {
                        splitIndices[i] = (int)arg;
                    }
                    else {
                        splitIndices[i] = (int) (instances.size() * arg) - 1;
                    }
                }
                if (splitIndices.length == 1) {
                    splitIndices = new int[]{0,splitIndices[0],instances.size() - 1};
                }
                BatchCompare.BatchCompare(outputFolder, instances, splitIndices, subsetLengths);
                break;
            case "stream":
                String[] windowSizesString = argv[2].split(",");
                int[] windowSizes = new int[windowSizesString.length];
                for (int i = 0; i < windowSizesString.length; i++) {
                    windowSizes[i] = Integer.parseInt(windowSizesString[i]);
                }
                instances = mergeInstances(allInstances);
                instances = discretizeDataSet(instances);
                DriftTimeline.DriftTimeline(outputFolder, instances, windowSizes, subsetLengths, true);
                break;
            case "stream_chunk":
                String[] arg2 = argv[2].split(",");
                int groupAttribute = Integer.parseInt(arg2[0]);
                int[] groupsSizes = new int[arg2.length - 1];
                for (int i = 1; i < arg2.length; i++) groupsSizes[i - 1] = Integer.parseInt(arg2[i]);
                //instances = mergeInstances(allInstances);
                DriftTimelineChunks.DriftTimelineChunks(outputFolder, allInstances, groupAttribute, groupsSizes, subsetLengths);
                break;
            case "moving_chunk":
                String[] arg2_m = argv[2].split(",");
                int groupAttribute_m = Integer.parseInt(arg2_m[0]);
                int[] groupsSizes_m = new int[arg2_m.length - 1];
                for (int i = 1; i < arg2_m.length; i++) groupsSizes_m[i - 1] = Integer.parseInt(arg2_m[i]);
                instances = mergeInstances(allInstances);
                DriftTimelineMovingChunks.DriftTimelineMovingChunks(outputFolder, instances, groupAttribute_m, groupsSizes_m, subsetLengths);
                break;
            case "stream_cont":
                String[] windowSizeString = argv[2].split(",");
                int[] windowSize = new int[windowSizeString.length];
                for (int i = 0; i < windowSizeString.length; i++) {
                    windowSize[i] = Integer.parseInt(windowSizeString[i]);
                }
                instances = mergeInstances(allInstances);
                instances = discretizeDataSet(instances);
                DriftTimeline.DriftTimeline(outputFolder, instances, windowSize, subsetLengths, false);
                break;
        }
    }

    static Instances mergeInstances(Instances[] allInstances) {
        Instances instances = new Instances(allInstances[0]);
        for (int i = 1; i < allInstances.length; i++) {
            instances.addAll(allInstances[i]);
        }
        return instances;
    }

    //TODO: Fix bug caused by deleting date attribute and model not accounting for it
    static int[] getAllAttributeSubsetLength(Instances instances) {
        int maxNumAtt = instances.numAttributes() - 1;
        for (int i = 0; i < instances.numAttributes(); i++) {
            if (instances.attribute(i).isDate() || instances.attribute(i).name().equals("date")) {
                maxNumAtt -= 1;
            }
        }
        int maxLength = Math.min(maxNumAtt, 3);
        int[] allLength = new int[maxLength];
        for (int i = 0; i < maxLength; i++) {
            allLength[i] = i + 1;
        }
        return allLength;
    }

    static String createFilePath(String[] folders) {
        String filename = folders[0];
        new File(filename).mkdir();
        for (int i = 1; i < folders.length; i++) {
            filename += "/" + folders[i];
            new File(filename).mkdir();
        }
        return filename;
    }

    static Instances loadDataSet(String filename) {
        try{
            // Check if any attribute is numeric
            Instances result;
            BufferedReader reader;

            reader = new BufferedReader(new FileReader(filename));
            result = new Instances(reader);
            reader.close();
            return result;
        }
        catch (Exception ex) {
            ex.printStackTrace();
            return new Instances("E", new ArrayList<Attribute>(), 0);
        }
    }

    static Instances discretizeDataSet(Instances dataSet) {
        try {
            ArrayList<Integer> continuousIndex = new ArrayList<>();
            for (int i = 0; i < dataSet.numAttributes(); i++) {
                continuousIndex.add(i);
            }
            int[] attIndex = new int[continuousIndex.size()];
            for (int i = 0; i < continuousIndex.size(); i++) attIndex[i] = continuousIndex.get(i);

            Discretize filter = new Discretize();
            filter.setUseEqualFrequency(true);
            filter.setBins(5);
            filter.setAttributeIndicesArray(attIndex);
            filter.setInputFormat(dataSet);

            Instances instances = Filter.useFilter(dataSet, filter);
            instances.setClassIndex(instances.numAttributes() - 1);
            return instances;
        }
        catch (Exception ex) {
            ex.printStackTrace();
            return new Instances("E", new ArrayList<Attribute>(), 0);
        }
    }

    public static Instances loadAnyDataSet(String filename) {
        Instances continuousData = loadDataSet(filename);
        if (filename.equals("./datasets/gas-sensor.arff")) {
            double[] classAttVals = continuousData.attributeToDoubleArray(0);
            Attribute classAtt = continuousData.attribute(0);
            continuousData.deleteAttributeAt(0);
            continuousData.insertAttributeAt(classAtt, continuousData.numAttributes());
            for (int i = 0; i < classAttVals.length; i++) {
                continuousData.get(i).setValue(continuousData.classIndex(), classAttVals[i]);
            }
        }
        Instances instances = discretizeDataSet(continuousData);
        instances.setClassIndex(instances.numAttributes() - 1);
        return instances;
    }

    public static int[] getAttributeIndicies(Instances instances) {
        ArrayList<Integer> indicesList = new ArrayList<>();
        for (int i = 0; i < instances.numAttributes() - 1; i++) {
            indicesList.add(i);
        }
        int[] attributeIndices = new int[indicesList.size()];
        for (int i = 0; i < indicesList.size(); i++) attributeIndices[i] = indicesList.get(i);
        return attributeIndices;
    }
}
