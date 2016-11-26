package main.models;

import org.apache.commons.lang3.ArrayUtils;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by loongkuan on 26/11/16.
 */
public class FrequencyMaps {
    int totalFrequency;
    ArrayList<HashMap<Integer, int[]>> frequencyMaps;
    int nAttributeSubsets;
    int nClasses;
    int[] attributesAvailable;
    int nAttributesActive;

    public FrequencyMaps(Instances initialInstances, int nAttributesActive, int[] attributesAvailable) {
        this.nAttributeSubsets = FrequencyMaps.nCr(initialInstances.numAttributes() - 1, nAttributesActive);
        this.nClasses = initialInstances.numClasses();
        this.attributesAvailable = attributesAvailable;
        this.nAttributesActive = nAttributesActive;

        this.frequencyMaps = new ArrayList<>();
        // Add a frequencyMap for each attribute subset
        for (int i = 0; i < nAttributesActive; i++) {
            this.frequencyMaps.add(new HashMap<>());
        }

        // Add initialInstances to frequencyMaps
        for (int i = 0; i < initialInstances.size(); i++) {
            this.addInstance(initialInstances.get(i));
        }
    }

    private void editInstance(Instance instance, int amount) {
        this.totalFrequency += amount;
        for (int j = 0; j < nAttributeSubsets; j++) {
            int[] activeAttributes = getKthCombination(j, attributesAvailable, nAttributesActive);
            int partialHash = FrequencyMaps.instanceToPartialHash(instance, activeAttributes);
            if (!this.frequencyMaps.get(j).containsKey(partialHash)) {
                this.frequencyMaps.get(j).put(partialHash, new int[nClasses + 1]);
            }
            int[] prevFreq = this.frequencyMaps.get(j).get(partialHash);
            prevFreq[0] += amount;
            prevFreq[(int)instance.classValue() + 1] += amount;
            this.frequencyMaps.get(j).replace(partialHash, prevFreq);
        }
    }

    public void addInstance(Instance instance) {
        this.editInstance(instance, 1);
    }

    public void removeInstance(Instance instance) {
        this.editInstance(instance, -1);
    }

    private static int[] getKthCombination(int k, int[] elements, int choices) {
        if (choices == 0) return new int[]{};
        else if (elements.length == choices) return  elements;
        else {
            int nCombinations = nCr(elements.length - 1, choices - 1);
            if (k < nCombinations) return ArrayUtils.addAll(ArrayUtils.subarray(elements, 0, 1),
                    getKthCombination(k, ArrayUtils.subarray(elements, 1, elements.length), choices - 1));
            else return getKthCombination(k - nCombinations, ArrayUtils.subarray(elements, 1, elements.length), choices);
        }
    }

    private static int nCr(int n, int r) {
        if (r >= n /2) r = n - r;
        int ans = 1;
        for (int i = 1; i <= r; i++) {
            ans *= n - r + i;
            ans /= i;
        }
        return ans;
    }

    private static Instance stripInstance(Instance instance, int[] activeAttributes) {
        Instance strippedInstance = new DenseInstance(instance);
        // Iterate over attributes in strippedInstance
        for (int i = 0; i < strippedInstance.numAttributes(); i++) {
            // If not class or active attribute set missing
            if (!ArrayUtils.contains(activeAttributes, i)) {
                strippedInstance.setMissing(i);
            }
        }
        return strippedInstance;
    }

    private static int instanceToPartialHash(Instance instance, int[] activeAttributes) {
        int hash = 0;
        int base = 1;
        // Iterate over attributes in strippedInstance
        for (int i = 0; i < instance.numAttributes() - 1; i++) {
            // If not class or active attribute set missing
            if (ArrayUtils.contains(activeAttributes, i)) {
                hash += base * (int)instance.value(i);
                base *= instance.attribute(i).numValues();
            }
        }
        return hash;
    }

    private static Instance partialHashToInstance(int partialHash, int[] activeAttributes, Instance exampleInst) {
        Instance partialInstance = new DenseInstance(exampleInst.numAttributes());

        for (int i = 0; i < partialInstance.numAttributes() - 1; i++) {
            if (ArrayUtils.contains(activeAttributes, i)) {
                partialInstance.setValue(i, partialHash % exampleInst.attribute(i).numValues());
                partialHash = partialHash / exampleInst.attribute(i).numValues();
            }
            else {
                partialInstance.setMissing(i);
            }
        }
        return partialInstance;
    }
}
