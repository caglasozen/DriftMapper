package main.models;

import org.apache.commons.lang3.ArrayUtils;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by loongkuan on 11/05/16.
 **/
public class NaiveMatrix {
    private Instances allInstances;
    private HashMap<Integer, int[]> frequencyTable;
    private HashMap<Integer, Integer> instanceSum;
    private int[][] attributeSum;       // Maps attribute index to a list of value frequency
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
        // Build attributeSum ragged array
        attributeSum = new int[dataSet.numAttributes()][];
        for (int i = 0; i < dataSet.numAttributes(); i++) {
            attributeSum[i] = new int[dataSet.attribute(i).numValues()];
        }
        for (Instance inst : dataSet) {
            int instHash = convertInstToHash(inst);
            int classHash = convertClassToHash(inst);
            // Attribute Sum
            for (int i = 0; i < inst.numAttributes(); i++) {
                attributeSum[i][(int)inst.value(i)] += 1;
            }
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
        return hash;
    }

    private Instance convertHashToInst(int hash) {
        DenseInstance instance = new DenseInstance(allInstances.get(0));
        instance.setDataset(allInstances);
        for (int i = instance.numAttributes() - 1; i >= 0; i--) {
            int current_value = 1;
            for (int j = i - 1; j >= 0; j--) {
                current_value *= this.allInstances.attribute(j).numValues();
            }
            int value = hash / current_value;
            value = value < instance.attribute(i).numValues() ? value : instance.attribute(i).numValues() - 1;
            instance.setValue(i, (double) value);
            hash = hash - (current_value * value);
        }
        return instance;
    }

    private static boolean isPartialInstance(Instance instance) {
        return instance.hasMissingValue();
    }

    private ArrayList<Integer> convertPartialInstToHashes(Instance instance) {
        DenseInstance copyInstance = new DenseInstance(instance);
        copyInstance.setDataset(instance.dataset());
        return(new ArrayList<>(convertPartialInstToHashes(copyInstance, 0, new HashSet<>())));
    }

    private HashSet<Integer> convertPartialInstToHashes(Instance instance,
                                                                 int currentAttributeIndex,
                                                                 HashSet<Integer> hashesSoFar) {
        if (currentAttributeIndex >= instance.numAttributes()) {
            hashesSoFar.add(convertInstToHash(instance));
            return(hashesSoFar);
        }
        else if (instance.isMissing(currentAttributeIndex)) {
            for (int attVal = 0; attVal < instance.attribute(currentAttributeIndex).numValues(); attVal++) {
                if (this.attributeSum[currentAttributeIndex][attVal] > 0) {
                    instance.setValue(currentAttributeIndex, (float)attVal);
                    hashesSoFar = convertPartialInstToHashes(instance, currentAttributeIndex + 1, hashesSoFar);
                    instance.setMissing(currentAttributeIndex);
                }
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
        return true;
    }

    private int getNumberFullInstance(Instance partialInstance) {
        int nFullInstance = 1;
        for (int i = 0; i < partialInstance.numAttributes(); i++) {
            if (partialInstance.isMissing(i)) {
                int nValues = 0;
                for (int j = 0; j < partialInstance.attribute(i).numValues(); j++) {
                    nValues += this.attributeSum[i][j] > 0 ? 1 : 0;
                }
                nFullInstance *= nValues;
            }
        }
        return nFullInstance;
    }

    private boolean isPartialInstancePossible(Instance partialInstance) {
        for (int i = 0; i < partialInstance.numAttributes(); i++) {
            if (!partialInstance.isMissing(i)) {
                if (!(this.attributeSum[i][(int)partialInstance.value(i)] > 0)) {
                    return false;
                }
            }
        }
        return true;
    }

    private int getPartialInstanceFrequency(Instance instance) {
        return this.getPartialInstanceFrequency(instance, false);
    }
    private int getPartialInstanceFrequency(Instance instance, boolean classSpecific) {
        if (!partialVectorExists(instance)) return 0;
        int totalFrequency = 0;
        int nFullInstance = getNumberFullInstance(instance);
        if (!isPartialInstancePossible(instance)) return 0;
        if (nFullInstance < this.allInstances.size()) {
            ArrayList<Integer> instanceHashes = convertPartialInstToHashes(instance);
            int classHash = convertClassToHash(instance);
            for (int hash : instanceHashes) {
                if (this.frequencyTable.containsKey(hash)) {
                    totalFrequency += classSpecific ? this.frequencyTable.get(hash)[classHash] : this.instanceSum.get(hash);
                }
            }
        }
        else {
            for (Instance dataInst : this.allInstances) {
                boolean isMatch = true;
                for (int j = 0; j < dataInst.numAttributes(); j++) {
                    isMatch = isMatch && (instance.isMissing(j) || dataInst.value(j) == instance.value(j));
                }
                totalFrequency += isMatch ? 1 : 0;
            }
        }
        return totalFrequency;
    }

    public double findPv(Instance instance) {
        return (double)this.getPartialInstanceFrequency(instance) / (double)this.sampleSize;
    }

    public double findPy(Instance instance) {
        int classHash= convertClassToHash(instance);
        return classHash != -1 ? (double)this.attributeSum[instance.classIndex()][classHash] / (double)this.sampleSize : 0.0f;
    }

    public double findPyGv(Instance instance) {
        int covariateFrequency = this.getPartialInstanceFrequency(instance);
        return covariateFrequency > 0 && convertClassToHash(instance) != -1 ?
                (double)this.getPartialInstanceFrequency(instance, true) / (double)covariateFrequency : 0.0f;
    }

    public double findPvGy(Instance instance) {
        if (convertClassToHash(instance) == -1) return 0.0f;
        int classFrequency = this.attributeSum[instance.classIndex()][convertClassToHash(instance)];
        return classFrequency > 0 ?
                (double) this.getPartialInstanceFrequency(instance, true) / (double) classFrequency : 0.0f;
    }
}
