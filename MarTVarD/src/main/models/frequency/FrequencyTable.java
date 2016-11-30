package main.models.frequency;

import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by loongkuan on 11/05/16.
 **/
public class FrequencyTable extends BaseFrequencyModel{
    private HashMap<Integer, int[]> frequencyTable;
    private int[][] attributeSum;       // Maps attribute index to a list of value frequency

    public FrequencyTable(Instances dataSet, int nAttributesActive, int[] attributesAvailable) {
        this.attributeSubsetLength = nAttributesActive;
        this.attributesAvailable = attributesAvailable;

        // Get bases for hash
        int base = 1;
        this.hashBases = new int[dataSet.numAttributes() - 1];
        for (int i = 0; i < dataSet.numAttributes() - 1; i++) {
            this.hashBases[i] = base;
            base *= dataSet.attribute(attributesAvailable[i]).numValues();
        }

        // Get sample instance
        this.allInstances = new Instances(dataSet);
        this.exampleInst = dataSet.firstInstance();

        // Save the sample size
        this.totalFrequency = dataSet.size();

        frequencyTable = new HashMap<>();
        // Build attributeSum ragged array
        attributeSum = new int[dataSet.numAttributes()][];
        for (int i = 0; i < dataSet.numAttributes(); i++) {
            attributeSum[i] = new int[dataSet.attribute(i).numValues()];
        }

        // Populate the Frequency Matrix and Sum tables
        for (Instance instance : dataSet) {
            this.addInstance(instance);
        }
    }

    @Override
    public void changeAttributeSubsetLength(int length) {
        this.attributeSubsetLength = length;
    }

    @Override
    public void addInstance(Instance instance) {
        int instHash = instanceToPartialHash(instance, this.attributesAvailable, hashBases);
        int classHash = (int)instance.classValue();
        // Attribute Sum
        for (int i = 0; i < instance.numAttributes(); i++) {
            attributeSum[i][(int)instance.value(i)] += 1;
        }

        if (!this.frequencyTable.containsKey(instHash)) {
            this.frequencyTable.put(instHash, new int[this.allInstances.numClasses() + 1]);
        }
        // Frequency Matrix
        int[] row = this.frequencyTable.get(instHash);
        row[classHash + 1] += 1;
        row[0] += 1;
        this.frequencyTable.put(instHash, row);
    }

    @Override
    public void removeInstance(Instance instance) {
        int instHash = instanceToPartialHash(instance, this.attributesAvailable,hashBases);
        int classHash = (int)instance.classValue();
        // Attribute Sum
        for (int i = 0; i < instance.numAttributes(); i++) {
            attributeSum[i][(int)instance.value(i)] -= 1;
        }

        // Frequency Matrix
        int[] row = this.frequencyTable.get(instHash);
        row[classHash + 1] -= 1;
        row[0] -= 1;
        this.frequencyTable.put(instHash, row);
        if (this.frequencyTable.get(instHash)[0] <= 0) {
            this.frequencyTable.remove(instHash);
        }
    }

    @Override
    protected Set<Integer> getAllHashes(int[] attributeSubset) {
        HashSet<Integer> hashes = new HashSet<>();
        for (int hash : this.frequencyTable.keySet()) {
            Instance instance = partialHashToInstance(hash, this.attributesAvailable, this.exampleInst, this.hashBases);
            int currentHash = instanceToPartialHash(instance, attributeSubset, hashBases);
            if (!hashes.contains(currentHash)) {
                hashes.add(currentHash);
            }
        }
        return hashes;
    }

    @Override
    protected int findFv(Instance instance, int[] attributesSubset) {
        return this.getPartialInstanceFrequency(instance, attributesSubset, false);
    }

    @Override
    protected int findFy(int classIndex) {
        return this.attributeSum[this.exampleInst.classIndex()][classIndex];
    }

    @Override
    protected int findFvy(Instance instance, int[] attributesSubset, int classIndex) {
        instance.setClassValue(classIndex);
        return this.getPartialInstanceFrequency(instance, attributesSubset, true);
    }

    private boolean partialVectorExists(Instance instance, int[] attributeSubset) {
        if (!this.allInstances.checkInstance(instance)) return false;
        for (int i : attributeSubset) {
            if (instance.value(i) >= this.allInstances.attribute(i).numValues()) return false;
        }
        return true;
    }

    private boolean isPartialInstancePossible(Instance instance, int[] attributeSubset) {
        for (int i : attributeSubset) {
            if (!(this.attributeSum[i][(int)instance.value(i)] > 0)) {
                return false;
            }
        }
        return true;
    }

    private int getPartialInstanceFrequency(Instance instance, int[] attributeSubset, boolean classSpecific) {
        if (!partialVectorExists(instance, attributeSubset)) return 0;
        if (!isPartialInstancePossible(instance, attributeSubset)) return 0;

        int instHash = instanceToPartialHash(instance, attributeSubset, hashBases);

        int totalFrequency = 0;
        for (int hash: this.frequencyTable.keySet()) {
            if (isPartialHashEqualHash(instHash, hash, attributeSubset, this.hashBases, this.exampleInst)) {
                totalFrequency += classSpecific ?
                        this.frequencyTable.get(hash)[(int)instance.classValue() + 1] :
                        this.frequencyTable.get(hash)[0];
            }
        }
        return totalFrequency;
    }
}
