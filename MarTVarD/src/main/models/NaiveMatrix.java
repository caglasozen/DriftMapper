package main.models;

import org.apache.commons.lang3.ArrayUtils;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Created by loongkuan on 11/05/16.
 **/
public class NaiveMatrix {
    private Instances allInstances;
    private HashMap<Integer, int[]> frequencyTable;
    private HashMap<Integer, Integer> instanceSum;
    private int[] classSum;
    private int sampleSize;

    public NaiveMatrix(Instances dataSet) {
        // Get sample instance
        this.allInstances = new Instances(dataSet);

        // Get number of classes
        int nClasses = dataSet.classAttribute().numValues();

        // Get number hashes/combinations
        int nCombinations = 1;
        for (int i = 0; i < dataSet.numAttributes(); i++) {
            if (i != dataSet.classIndex()) {
                nCombinations *= dataSet.attribute(i).numValues();
            }
        }

        // Save the sample size
        this.sampleSize = dataSet.size();

        // Populate the Frequency Matrix and Sum tables
        frequencyTable = new HashMap<>();
        instanceSum = new HashMap<>();
        classSum = new int[nClasses];
        for (Instance inst : dataSet) {
            int instHash = convertInstToHash(inst);
            int classHash = convertClassToHash(inst);
            // Class Sum
            this.classSum[classHash] += 1;
            // Check if the instance hash is registered in the lookup tables
            if (!this.vectorExists(inst)) {
                this.instanceSum.put(instHash, 0);
                this.frequencyTable.put(instHash, new int[nClasses]);
            }
            // Instance Sum
            this.instanceSum.put(instHash, this.instanceSum.get(instHash) + 1);
            // Frequency Matrix
            this.frequencyTable.get(instHash)[classHash] += 1;
        }
    }

    public Instances getAllInstances() {
        return allInstances;
    }

    private static int convertInstToHash(Instance instance) {
        int hash = 0;
        for (int i = 0; i < instance.numAttributes(); i++) {
            // Do not include the class in hash
            if (i != instance.classIndex()) {
                int current_hash = 1;
                // Get weight of current digit
                for (int j = 0; j < i; j++) {
                    // Do not include class in hash
                    if (j != instance.classIndex())
                        current_hash *= instance.attribute(j).numValues();
                }
                // Multiply weight of current digit with value of current digit
                current_hash *= (int)instance.value(i);
                // Add weighted value of current digit to hash
                hash += current_hash;
            }
        }
        return hash;
    }

    private static boolean isPartialInstance(Instance instance) {
        return instance.hasMissingValue();
    }

    private static ArrayList<Integer> convertPartialInstToHashes(Instance instance) {
        return(convertPartialInstToHashes(instance, 0, new ArrayList<>()));
    }

    private static ArrayList<Integer> convertPartialInstToHashes(Instance instance,
                                                                 int currentAttributeIndex,
                                                                 ArrayList<Integer> hashesSoFar) {
        if (currentAttributeIndex == instance.numAttributes() - 1) {
            hashesSoFar.add(convertInstToHash(instance));
            return(hashesSoFar);
        }
        else if (instance.isMissing(currentAttributeIndex)) {
            for (int attVal = 0; attVal < instance.attribute(currentAttributeIndex).numValues(); attVal++) {
                instance.setValue(currentAttributeIndex, (float)attVal);
                hashesSoFar = convertPartialInstToHashes(instance, currentAttributeIndex + 1, hashesSoFar);
            }
            return(hashesSoFar);
        }
        else {
            hashesSoFar = convertPartialInstToHashes(instance, currentAttributeIndex + 1, hashesSoFar);
            return(hashesSoFar);
        }
    }

    private static int convertClassToHash(Instance instance) {
        return (int)instance.classValue();
    }

    private boolean vectorExists(Instance instance) {
        boolean exists = this.allInstances.checkInstance(instance);
        for (int i = 0; i < instance.numAttributes(); i++) {
            exists = exists && instance.value(i) < this.allInstances.attribute(i).numValues();
        }
        Integer covariateHash = convertInstToHash(instance);
        return exists && this.frequencyTable.containsKey(covariateHash);
    }

    private boolean partialVectorExists(Instance instance) {
        if (!this.allInstances.checkInstance(instance)) return false;
        for (int i = 0; i < instance.numAttributes(); i++) {
            if (!instance.isMissing(i) && instance.value(i) >= this.allInstances.attribute(i).numValues()) return false;
        }
        Integer covariateHash = convertInstToHash(instance);
        return this.frequencyTable.containsKey(covariateHash);
    }

    private int getPartialInstanceFrequency(Instance instance) {
        return this.getPartialInstanceFrequency(instance, false);
    }
    private int getPartialInstanceFrequency(Instance instance, boolean classSpecific) {
        if (!partialVectorExists(instance)) return 0;
        int totalFrequency = 0;
        ArrayList<Integer> instanceHashes = convertPartialInstToHashes(instance);
        int classHash = convertClassToHash(instance);
        for (int hash : instanceHashes) {
            if (this.frequencyTable.containsKey(hash)) {
                totalFrequency += classSpecific ? this.frequencyTable.get(hash)[classHash] : this.instanceSum.get(hash);
            }
        }
        return totalFrequency;
    }

    public double findPv(Instance instance) {
        return (double)this.getPartialInstanceFrequency(instance) / (double)this.sampleSize;
    }

    public double findPy(Instance instance) {
        int classHash= convertClassToHash(instance);
        return classHash != -1 ? (double)this.classSum[classHash] / (double)this.sampleSize : 0.0f;
    }

    public double findPyGv(Instance instance) {
        int covariateFrequency = this.getPartialInstanceFrequency(instance);
        return covariateFrequency > 0 && convertClassToHash(instance) != -1 ?
                (double)this.getPartialInstanceFrequency(instance, true) / (double)covariateFrequency : 0.0f;
    }

    public double findPvGy(Instance instance) {
        if (convertClassToHash(instance) == -1) return 0.0f;
        int classFrequency = this.classSum[convertClassToHash(instance)];
        return classFrequency > 0 ?
                (double) this.getPartialInstanceFrequency(instance, true) / (double) classFrequency : 0.0f;
    }
}
