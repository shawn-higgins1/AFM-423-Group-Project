	末想Oe8@末想Oe8@!末想Oe8@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-末想Oe8@ﾁ鵠-5@1ﾌ(房Zﾏ?A_)ﾋﾇｺｨ?I�"nN%@*	     H@2U
Iterator::Model::ParallelMapV2ﾌ]Kﾈ=�?!VUUUUｵ;@)ﾌ]Kﾈ=�?1VUUUUｵ;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatﾌ]Kﾈ=�?!VUUUUｵ;@)M�惨t$�?1ｫｪｪｪｪ�7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateZd;ﾟO合?!VUUUU�7@)ﾒ ﾞ	�?1ｫｪｪｪｪ
0@:Preprocessing2F
Iterator::ModelUﾁｨ､N@�?!UUUUU匹@)ｺI+�v?1ｪｪｪｪｪ�&@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceﾅ�1w-!o?!ｪｪｪｪｪｪ@)ﾅ�1w-!o?1ｪｪｪｪｪｪ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�<,ﾔ壽�?!ｪｪｪｪｪjN@)-C��6j?1ｫｪｪｪｪｪ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�ｩ�ﾒMb`?!ｪｪｪｪｪｪ@)�ｩ�ﾒMb`?1ｪｪｪｪｪｪ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapｦ
F%u�?!UUUUUu:@)a2U0*ｩS?1      @:Preprocessing:ｪ
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
ﾅData preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
ﾒReading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
ﾅReading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
ｺOther data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisｧ
bothｫYour program is POTENTIALLY input-bound because 86.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"ﾌ12.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIXﾟ�[ﾀX@Q�S�?ﾒ�?Zno>Look at Section 3 for the breakdown of input time on the host.Bﾅ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ﾁ鵠-5@ﾁ鵠-5@!ﾁ鵠-5@      ��!       "	ﾌ(房Zﾏ?ﾌ(房Zﾏ?!ﾌ(房Zﾏ?*      ��!       2	_)ﾋﾇｺｨ?_)ﾋﾇｺｨ?!_)ﾋﾇｺｨ?:	�"nN%@�"nN%@!�"nN%@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qXﾟ�[ﾀX@y�S�?ﾒ�?