package main.run;

import main.DriftMeasurement;
import main.analyse.timeline.NaiveChunk;
import main.models.Model;
import main.models.frequency.FrequencyMaps;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by loongkuan on 20/12/2016.
 */
public class DriftTimelineChunks extends main{

    //TODO: Bug due to discretizing date attributes
    public static void DriftTimelineChunks(String resultFolder, Instances[] allSepInstances,
                                           int groupAttribute, int[] groupSizes, int[] subsetLengths) {
        Instances[] allGroupedInstances;
        if (groupAttribute >= 0) {
            Instances allInstances = mergeInstances(allSepInstances);
            ArrayList<Instances> groupedInstances = new ArrayList<>();
            double prevAttributeValue = -1;
            double[] groupedAttVals = allInstances.attributeToDoubleArray(groupAttribute);
            allInstances = discretizeDataSet(allInstances);
            allInstances.deleteAttributeAt(groupAttribute);
            for (int j = 0; j < allInstances.size(); j++) {
                Instance instance = allInstances.get(j);
                if (groupedAttVals[j] != prevAttributeValue) {
                    groupedInstances.add(new Instances(allInstances, 0));
                    prevAttributeValue = groupedAttVals[j];
                }
                groupedInstances.get(groupedInstances.size() - 1).add(instance);
            }
            allGroupedInstances = groupedInstances.toArray(new Instances[groupedInstances.size()]);
        }
        else {
            int[] sizes = new int[allSepInstances.length];
            for (int i = 0; i < allSepInstances.length; i++) {
                sizes[i] = allSepInstances[i].size();
            }
            Instances allInstances2 = mergeInstances(allSepInstances);
            allInstances2 = discretizeDataSet(allInstances2);
            allGroupedInstances = new Instances[allSepInstances.length];
            int currentIndex = 0;
            for (int i = 0; i < sizes.length; i++) {
                allGroupedInstances[i] = new Instances(allInstances2, currentIndex, sizes[i]);
                currentIndex += sizes[i];
            }
            groupSizes = new int[]{1};
        }

        for (int subsetLength : subsetLengths) {
            for (int groupSize : groupSizes) {
                runExperiment(allGroupedInstances, subsetLength, groupSize, resultFolder);
            }
        }
    }

    private static void runExperiment(Instances[] allInstances, int subsetLength, int groupSize, String resultFolder) {
        // Create new chunks of groupSize
        Instances[] newAllInstances = new Instances[(allInstances.length) / groupSize];
        for (int i = 0; i < (allInstances.length - (allInstances.length % groupSize)); i++) {
            if (i % groupSize == 0) {
                newAllInstances[i / groupSize] = new Instances(allInstances[i]);
            }
            else {
                newAllInstances[i / groupSize].addAll(allInstances[i]);
            }
        }
        int[] attributeIndices = getAttributeIndicies(newAllInstances[0]);

        Model referenceModel = new FrequencyMaps(newAllInstances[0], subsetLength, attributeIndices);
        NaiveChunk naiveChunk = new NaiveChunk(newAllInstances, DriftMeasurement.values(), referenceModel);

        System.out.println("\rDone test of Subset Length = " + subsetLength + " and chunk size of " + groupSize);

        for (DriftMeasurement driftMeasurement : DriftMeasurement.values()) {
            String file = resultFolder + "/" + driftMeasurement.name() + "_" + groupSize + "_" + subsetLength + ".csv";
            naiveChunk.writeResultsToFile(file, driftMeasurement);
        }
    }
}
