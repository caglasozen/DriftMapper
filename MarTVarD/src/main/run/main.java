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
    /**
     * single subsetLength1,subsetLength2,... splitProportion folder file1 file2 ...
     * single subsetLength1,subsetLength2,... splitIndex folder file1 file2 ...
     * multi subsetLength1,subsetLength2,... folder file1 file2 ...
     * stream subsetLength1,subsetLength2,... windowSize1,windowSize2,... folder file1 file2 file3 ...
     * @param argv experimentType folder file1 file2 file3 ...
     */
    public static void main(String[] argv) {
        //argv = new String[]{"single", "1,2,3", "0.5", "train_seed", "20130505", "20131129"};
        //argv = new String[]{"single", "1,2,3",  "0.5", "", "elecNormNew"};
        argv = new String[]{"stream", "1,2,3,4",  "48,336,1461,17520", "", "elecNormNew"};
        //argv = new String[]{"stream", "1,2,3",  "6171,43197,187845", "data_uni_antwerp", "water_2015"};
        //argv = new String[]{"multi", "1,2,3", "train_seed", "20130419", "20130505", "20130521", "20130606", "20130622"};

        String[] subsetLengthsString = argv[1].split(",");
        int[] subsetLengths = new int[subsetLengthsString.length];
        for (int i = 0; i < subsetLengthsString.length; i++) {
            subsetLengths[i] = Integer.parseInt(subsetLengthsString[i]);
        }

        String folder = "";
        String filename_comb = "";
        Instances allInstances;
        if (!argv[0].equals("multi")) {
            folder = argv[3];
            String[] files = ArrayUtils.subarray(argv, 4, argv.length);
            allInstances = loadAnyDataSet("./datasets/" + folder + "/" + files[0] + ".arff");
            filename_comb = files[0];
            for (int i = 1; i < files.length; i++) {
                allInstances.addAll(loadAnyDataSet("./datasets/" + folder + "/" + files[i] + ".arff"));
                filename_comb += "_" + files[i];
            }
        }
        else {
            folder = argv[2];
            String[] files = ArrayUtils.subarray(argv, 3, argv.length);
            allInstances = loadAnyDataSet("./datasets/" + folder + "/" + files[0] + ".arff");
            filename_comb = files[0];
            int windowSize = allInstances.size();
            for (int i = 1; i < files.length; i++) {
                System.out.println("Reading " + files[i]);
                allInstances.addAll(loadAnyDataSet("./datasets/" + folder + "/" + files[i] + ".arff"));
                filename_comb += "_" + files[i];
            }
            String filepath = getFilePath("./data_out", folder, "", "stream");
            DriftTimeline.DriftTimeline(filepath, allInstances, new int[]{windowSize}, subsetLengths);
        }

        String filepath;
        switch (argv[0]){
            case "single":
                filepath = getFilePath("./data_out", folder, filename_comb, "FrequencyMaps");
                double splitArg = Double.parseDouble(argv[2]);
                int splitIndex = splitArg < 1 && splitArg > 0 ? (int)(allInstances.size() * splitArg) : (int) splitArg;
                BatchCompare.BatchComapare(filepath, allInstances, splitIndex, subsetLengths);
                break;
            case "stream":
                filepath = getFilePath("./data_out", folder, filename_comb, "stream");
                String[] windowSizesString = argv[2].split(",");
                int[] windowSizes = new int[windowSizesString.length];
                for (int i = 0; i < windowSizesString.length; i++) {
                    windowSizes[i] = Integer.parseInt(windowSizesString[i]);
                }
                DriftTimeline.DriftTimeline(filepath, allInstances, windowSizes, subsetLengths);
        }
    }

    static int[] getAllAttributeSubsetLength(Instances instances) {
        int maxLength = Math.min(instances.numAttributes(), 3);
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
            if (dataSet.attribute(i).isNumeric()) continuousIndex.add(i);
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
}
