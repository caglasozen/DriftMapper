package main.run;

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
    //TODO: analyse subsetLength1,subsetLength2,... startIndex,MiddleIndex,EndIndex folder file1 file2 ...
    /**
     * analyse      subsetLength1,subsetLength2,... splitStart,splitMiddle,splitEnd             folder file1 file2 ...
     * analyse      subsetLength1,subsetLength2,... startPropostion,Middle,End                  folder file1 file2 ...
     * stream       subsetLength1,subsetLength2,... windowSize1,windowSize2,...                 folder file1 file2 ...
     * stream_cont  subsetLength1,subsetLength2,... windowSize1,windowSize2,...                 folder file1 file2 ...
     * stream_chunk subsetLength1,subsetLength2,... groupAttIndex,groupSize1,groupsSize2,...    folder file1 file2 ...
     * @param argv experimentType folder file1 file2 file3 ...
     */
    public static void main(String[] argv) {
        //argv = new String[]{"analyse", "1,2,3", "0.5", "train_seed", "20130505", "20131129"};
        //argv = new String[]{"analyse", "1,2,3",  "0.5", "", "elecNormNew"};
        //argv = new String[]{"analyse", "1,2",  "17532", "", "elecNormNew"};
        //argv = new String[]{"analyse", "1,2",  "240796", "", "airlines"};
        //argv = new String[]{"stream", "1,2",  "48,336,1461,17520", "", "elecNormNew"};
        //argv = new String[]{"stream", "1,2,3",  "6048,42336,183859", "data_uni_antwerp", "water_2015"};
        //argv = new String[]{"stream", "1,2,3,4",  "10000,50000,100000,500000", "", "sensor"};
        //argv = new String[]{"stream_chunk", "1,2,3",  "-1", "train_seed", "20130419", "20130505", "20130521", "20130606", "20130622"};
        //argv = new String[]{"stream_chunk", "1,2,3,4",  "4,1,7", "", "airlines"};
        //argv = new String[]{"stream_chunk", "1,2",  "2,1,7,30", "data_uni_antwerp", "water_2015"};
        //argv = new String[]{"stream", "1,2,3",  "10000,50000,100000,500000", "synthetic_5Att_5Val", "n1000000_m0.7_both"};
        //argv = new String[]{"stream", "1,2,3",  "133332,222221,399999,666666", "synthetic_5Att_5Val", "n1000000_m0.7_both"};
        argv = new String[]{"stream_cont", "1,2,3",  "10000,50000,100000,500000", "synthetic_5Att_5Val", "n1000000_m0.7_both"};
        //argv = new String[]{"stream", "1",  "500,1000,5000", "", "gas-sensor"};

        // Obtain Subset Length
        String[] subsetLengthsString = argv[1].split(",");
        int[] subsetLengths = new int[subsetLengthsString.length];
        for (int i = 0; i < subsetLengthsString.length; i++) {
            subsetLengths[i] = Integer.parseInt(subsetLengthsString[i]);
        }

        // Obtain information regarding location of data and result output directory
        String folder = argv[3];
        String[] files = ArrayUtils.subarray(argv, 4, argv.length);
        Instances[] allInstances = new Instances[files.length];
        String filename_comb = files[0];
        for (int i = 0; i < files.length; i++) {
            allInstances[i] = loadAnyDataSet("./datasets/" + folder + "/" + files[i] + ".arff");
            filename_comb += i > 0 ? "_" + files[i] : "";
        }
        String filepath = getFilePath("./data_out", folder, filename_comb, argv[0]);

        switch (argv[0]){
            case "analyse":
                String[] splitArgs = argv[2].split(",");
                Instances instances = mergeInstances(allInstances);
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
                BatchCompare.BatchCompare(filepath, instances, splitIndices, subsetLengths);
                break;
            case "stream":
                String[] windowSizesString = argv[2].split(",");
                int[] windowSizes = new int[windowSizesString.length];
                for (int i = 0; i < windowSizesString.length; i++) {
                    windowSizes[i] = Integer.parseInt(windowSizesString[i]);
                }
                DriftTimeline.DriftTimeline(filepath, mergeInstances(allInstances), windowSizes, subsetLengths, true);
                break;
            case "stream_chunk":
                String[] arg2 = argv[2].split(",");
                int groupAttribute = Integer.parseInt(arg2[0]);
                int[] groupsSizes = new int[arg2.length - 1];
                for (int i = 1; i < arg2.length; i++) groupsSizes[i - 1] = Integer.parseInt(arg2[i]);
                DriftTimelineChunks.DriftTimelineChunks(filepath, allInstances, groupAttribute, groupsSizes, subsetLengths);
                break;
            case "stream_cont":
                String[] windowSizeString = argv[2].split(",");
                int[] windowSize = new int[windowSizeString.length];
                for (int i = 0; i < windowSizeString.length; i++) {
                    windowSize[i] = Integer.parseInt(windowSizeString[i]);
                }
                DriftTimeline.DriftTimeline(filepath, mergeInstances(allInstances), windowSize, subsetLengths, false);
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

    static String getFilePath(String resultDir, String dataDir, String dataFileName, String experimentName) {
        String filname = resultDir;
        new File(filname).mkdir();
        filname += dataDir.equals("") ? "" : "/"  + dataDir;
        new File(filname).mkdir();
        filname += "/" + dataFileName;
        new File(filname).mkdir();
        filname += "/" + experimentName;
        new File(filname).mkdir();
        return filname;
    }

    static Instances loadDataSet(String filename) throws IOException {
        // Check if any attribute is numeric
        Instances result;
        BufferedReader reader;

        reader = new BufferedReader(new FileReader(filename));
        result = new Instances(reader);
        reader.close();
        return result;
    }

    private static Instances discretizeDataSet(Instances dataSet) throws Exception{
        ArrayList<Integer> continuousIndex = new ArrayList<>();
        for (int i = 0; i < dataSet.numAttributes(); i++) {
            if (dataSet.attribute(i).isNumeric() &&
                    !dataSet.attribute(i).isDate() &&
                    !dataSet.attribute(i).name().equals("date")) {
                continuousIndex.add(i);
            }
        }
        int[] attIndex = new int[continuousIndex.size()];
        for (int i = 0; i < continuousIndex.size(); i++) attIndex[i] = continuousIndex.get(i);

        Discretize filter = new Discretize();
        filter.setUseEqualFrequency(true);
        filter.setBins(5);
        filter.setAttributeIndicesArray(attIndex);
        filter.setInputFormat(dataSet);

        return Filter.useFilter(dataSet, filter);
    }

    public static Instances loadAnyDataSet(String filename) {
        try {
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
        catch (Exception ex) {
            ex.printStackTrace();
            return new Instances("E", new ArrayList<Attribute>(), 0);
        }
    }

    public static int[] getAttributeIndicies(Instances instances) {
        ArrayList<Integer> indicesList = new ArrayList<>();
        for (int i = 0; i < instances.numAttributes() - 1; i++) {
            if (!instances.attribute(i).isDate() && !instances.attribute(i).name().equals("date")) {
                indicesList.add(i);
            }
        }
        int[] attributeIndices = new int[indicesList.size()];
        for (int i = 0; i < indicesList.size(); i++) attributeIndices[i] = indicesList.get(i);
        return attributeIndices;
    }
}
