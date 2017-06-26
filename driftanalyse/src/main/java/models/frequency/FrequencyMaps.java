package models.frequency;

import models.Model;
import weka.core.Instance;
import weka.core.Instances;

import java.math.BigInteger;
import java.util.*;

/**
 * Created by loongkuan on 26/11/16.
 **/

public class FrequencyMaps extends BaseFrequencyModel {
    // Hash of attribute subset is key because array cannot be key
    private HashMap<BigInteger, HashMap<BigInteger, int[]>> frequencyMaps;
    private int[] classFreq;

    public FrequencyMaps(Instances dataset, int attributeSubsetLength, int[] attributesAvailable) {
        this.attributesAvailable = attributesAvailable;
        this.attributeSubsetLength = attributeSubsetLength;

        this.setDataset(dataset);
    }

    @Override
    public void reset() {
        this.allInstances = new Instances(this.allInstances, this.allInstances.size());
        this.changeAttributeSubsetLength(this.attributeSubsetLength);
    }

    @Override
    public Model copy() {
        return new FrequencyMaps(new Instances(this.allInstances, this.allInstances.size()),
                this.attributeSubsetLength, this.attributesAvailable);
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
        // New class frequency array
        this.classFreq = new int[this.allInstances.numClasses()];

        // Add initialInstances to frequencyMaps
        for (int i = 0; i < this.allInstances.size(); i++) {
            this.editInstance(this.allInstances.get(i), 1);
        }
    }

    @Override
    public void addInstance(Instance instance) {
        this.allInstances.add(instance);
        this.editInstance(instance, 1);
    }

    @Override
    public Instance removeInstance(int index) {
        Instance instance = this.allInstances.remove(index);
        this.editInstance(instance, -1);
        return instance;
    }

    @Override
    protected Set<BigInteger> getAllHashes(int[] attributeSubset) {
        return this.frequencyMaps.get(attributeSubsetToHash(attributeSubset)).keySet();
    }

    @Override
    public int findFv(Instance instance, int[] attributesSubset) {
        BigInteger hash = this.instanceToPartialHash(instance, attributesSubset);
        return !hashSeen(hash, attributesSubset) ? 0 : this.frequencyMaps.get(attributeSubsetToHash(attributesSubset)).get(hash)[0];
    }

    @Override
    public int findFy(int classIndex) {
        return this.classFreq[classIndex];
    }

    @Override
    public int findFvy(Instance instance, int[] attributesSubset, int classIndex) {
        BigInteger hash = this.instanceToPartialHash(instance, attributesSubset);
        return !hashSeen(hash, attributesSubset) ? 0 : this.frequencyMaps.get(attributeSubsetToHash(attributesSubset)).get(hash)[1 + classIndex];
    }

    private boolean hashSeen(BigInteger hash, int[] attributeSubset) {
        return this.frequencyMaps.get(attributeSubsetToHash(attributeSubset)).containsKey(hash);
    }

    private void editInstance(Instance instance, int amount) {
        this.classFreq[(int)instance.classValue()] += amount;
        int nAttributeSubsets = nCr(this.attributesAvailable.length, this.attributeSubsetLength);
        for (int j = 0; j < nAttributeSubsets; j++) {
            int[] activeAttributes = getKthCombination(j, attributesAvailable, attributeSubsetLength);
            BigInteger partialHash = this.instanceToPartialHash(instance, activeAttributes);
            BigInteger subsetHash = attributeSubsetToHash(activeAttributes);
            if (!this.frequencyMaps.get(subsetHash).containsKey(partialHash)) {
                this.frequencyMaps.get(subsetHash).put(partialHash, new int[1 + this.allInstances.numClasses()]);
            }
            int[] prevFreq = this.frequencyMaps.get(subsetHash).get(partialHash);
            prevFreq[0] += amount;
            prevFreq[1 + (int)instance.classValue()] += amount;
            this.frequencyMaps.get(subsetHash).replace(partialHash, prevFreq);
        }
    }


    /*
    @Override
    public double peakPosteriorDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        ArrayList<BigInteger> allHashes = mergeHashes((BaseFrequencyModel)modelToCompare, attributeSubset);
        BigInteger subsetHash = attributeSubsetToHash(attributeSubset);
        int nClasses = this.allInstances.numClasses();

        double dist = 0.0f;
        for (BigInteger hash : allHashes) {
            int pN = this.frequencyMaps.get(subsetHash).containsKey(hash) ?
                    this.frequencyMaps.get(subsetHash).get(hash)[0] : 0;
            int qN = ((FrequencyMaps)modelToCompare).frequencyMaps.get(subsetHash).containsKey(hash) ?
                    ((FrequencyMaps)modelToCompare).frequencyMaps.get(subsetHash).get(hash)[0] : 0;
            double weight = (((double)pN / (double)this.allInstances.size()) +
                    ((double)qN / (double)((FrequencyMaps)modelToCompare).allInstances.size())) / 2;
            double currentDist = 0.0f;

            int[] pFreqs = this.frequencyMaps.get(subsetHash).containsKey(hash) ?
                    this.frequencyMaps.get(subsetHash).get(hash) : new int[nClasses + 1];
            int[] qFreqs = ((FrequencyMaps)modelToCompare).frequencyMaps.get(subsetHash).containsKey(hash) ?
                    ((FrequencyMaps)modelToCompare).frequencyMaps.get(subsetHash).get(hash) : new int[1 + nClasses];

            for (int classIndex = 0; classIndex < this.allInstances.numClasses(); classIndex++) {
                double tmpDist = pN == 0 ? 0 : (double)pFreqs[1 + classIndex] / (double)pN;
                tmpDist -= qN == 0 ? 0 : ((double)qFreqs[1 + classIndex] / (double)qN);
                currentDist += Math.abs(tmpDist);
            }
            dist += (currentDist / (double)2) * weight;
        }
        return dist;
    }
    */
}
