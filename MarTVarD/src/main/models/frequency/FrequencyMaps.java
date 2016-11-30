package main.models.frequency;

import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Created by loongkuan on 26/11/16.
 **/

public class FrequencyMaps extends BaseFrequencyModel {
    // Hash of attribute subset is key because array cannot be key
    private HashMap<Integer, HashMap<Integer, int[]>> frequencyMaps;
    private int[] classFreq;
    private HashMap<int[], HashMap<Integer, Integer>> covariateFreq;

    public FrequencyMaps(int attributeSubsetLength, int[] attributesAvailable) {
        this.attributesAvailable = attributesAvailable;
    }

    public FrequencyMaps(Instances initialInstances, int attributeSubsetLength, int[] attributesAvailable) {
        this.attributesAvailable = attributesAvailable;
        this.attributeSubsetLength = attributeSubsetLength;
        this.exampleInst = initialInstances.firstInstance();
        this.classFreq = new int[this.exampleInst.numClasses()];
        this.covariateFreq = new HashMap<>();
        this.allInstances = initialInstances;

        // Get bases for hash
        int base = 1;
        this.hashBases = new int[initialInstances.numAttributes() - 1];
        for (int i = 0; i < initialInstances.numAttributes() - 1; i++) {
            this.hashBases[i] = base;
            base *= initialInstances.attribute(attributesAvailable[i]).numValues();
        }

        this.changeAttributeSubsetLength(attributeSubsetLength);
    }

    @Override
    public void changeAttributeSubsetLength(int length) {
        this.attributeSubsetLength = length;

        int nAttributeSubsets = FrequencyMaps.nCr(this.attributesAvailable.length, this.attributeSubsetLength);
        // New frequency map
        this.frequencyMaps = new HashMap<>();
        // Add a frequencyMap for each attribute subset
        for (int i = 0; i < nAttributeSubsets; i++) {
            this.frequencyMaps.put(
                    attributeSubsetToHash(getKthCombination(i, this.attributesAvailable, this.attributeSubsetLength)),
                    new HashMap<>());
        }
        // Add initialInstances to frequencyMaps
        //TODO: Add fancy x out of n instances added printout
        int percentage = -1;
        for (int i = 0; i < this.allInstances.size(); i++) {
            //if (i % 10000 == 0) System.out.print("\rAdded " + i + " Instances out of " + this.allInstances.size());
            if (percentage != (int)((i/(double)this.allInstances.size()) * 100)) {
                percentage = (int)((i/(double)this.allInstances.size()) * 100);
                System.out.print("\rAdded " + percentage + "% of Instances ");
            }
            this.editInstance(this.allInstances.get(i), 1);
        }
    }

    @Override
    public void addInstance(Instance instance) {
        this.allInstances.add(instance);
        this.editInstance(instance, 1);
    }

    @Override
    public void removeInstance(Instance instance) {
        this.allInstances.remove(instance);
        this.editInstance(instance, -1);
    }

    @Override
    protected Set<Integer> getAllHashes(int[] attributeSubset) {
        return this.frequencyMaps.get(attributeSubsetToHash(attributeSubset)).keySet();
    }

    @Override
    protected int findFv(Instance instance, int[] attributesSubset) {
        int hash = this.instanceToPartialHash(instance, attributesSubset);
        return !hashSeen(hash, attributesSubset) ? 0 : this.frequencyMaps.get(attributeSubsetToHash(attributesSubset)).get(hash)[0];
    }

    @Override
    protected int findFy(int classIndex) {
        return this.classFreq[classIndex];
    }

    @Override
    protected int findFvy(Instance instance, int[] attributesSubset, int classIndex) {
        int hash = this.instanceToPartialHash(instance, attributesSubset);
        return !hashSeen(hash, attributesSubset) ? 0 : this.frequencyMaps.get(attributeSubsetToHash(attributesSubset)).get(hash)[1 + classIndex];
    }

    private boolean hashSeen(int hash, int[] attributeSubset) {
        return this.frequencyMaps.get(attributeSubsetToHash(attributeSubset)).containsKey(hash);
    }

    private void editInstance(Instance instance, int amount) {
        this.totalFrequency += amount;
        this.classFreq[(int)instance.classValue()] += amount;
        int nAttributeSubsets = nCr(this.attributesAvailable.length, this.attributeSubsetLength);
        for (int j = 0; j < nAttributeSubsets; j++) {
            int[] activeAttributes = getKthCombination(j, attributesAvailable, attributeSubsetLength);
            int partialHash = this.instanceToPartialHash(instance, activeAttributes);
            int subsetHash = attributeSubsetToHash(activeAttributes);
            if (!this.frequencyMaps.get(subsetHash).containsKey(partialHash)) {
                this.frequencyMaps.get(subsetHash).put(partialHash, new int[1 + this.exampleInst.numClasses()]);
            }
            int[] prevFreq = this.frequencyMaps.get(subsetHash).get(partialHash);
            prevFreq[0] += amount;
            prevFreq[1 + (int)instance.classValue()] += amount;
            this.frequencyMaps.get(subsetHash).replace(partialHash, prevFreq);
        }
    }

    private static int attributeSubsetToHash(int[] attributeSubset) {
        int hash = 0;
        for (int i : attributeSubset) {
            hash += Math.pow(2, i);
        }
        return hash;
    }
}
