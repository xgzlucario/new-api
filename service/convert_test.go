package service

import (
	"testing"

	"github.com/QuantumNous/new-api/dto"
	"github.com/stretchr/testify/require"
)

// 反向转换：OpenAI usage → Anthropic ClaudeUsage。
// 关键不变量：Anthropic 协议里 input_tokens 不含 cache，所以要从 OpenAI 的 prompt_tokens
// 把 cached_tokens + cache_creation 全部减回去。与 buildOpenAIStyleUsageFromClaudeUsage 对称。
func TestBuildClaudeUsageFromOpenAIUsage_InputExcludesCache(t *testing.T) {
	// 模拟用户线上看到的样本：上游 OpenAI 协议返回的 prompt_tokens 已经把 cache 滚进去
	// 之前的 bug 是直接抄成 InputTokens=4847，导致下游审计统计双计 4736
	oai := &dto.Usage{
		PromptTokens:     4847,
		CompletionTokens: 29,
		PromptTokensDetails: dto.InputTokenDetails{
			CachedTokens:         4736,
			CachedCreationTokens: 0,
		},
	}

	got := buildClaudeUsageFromOpenAIUsage(oai)
	require.NotNil(t, got)
	require.Equal(t, 111, got.InputTokens, "input_tokens 必须不含 cache_read")
	require.Equal(t, 29, got.OutputTokens)
	require.Equal(t, 4736, got.CacheReadInputTokens)
	require.Equal(t, 0, got.CacheCreationInputTokens)
	require.Nil(t, got.CacheCreation)
}

func TestBuildClaudeUsageFromOpenAIUsage_InputExcludesCacheCreation(t *testing.T) {
	oai := &dto.Usage{
		PromptTokens:     1500,
		CompletionTokens: 50,
		PromptTokensDetails: dto.InputTokenDetails{
			CachedTokens:         200,
			CachedCreationTokens: 300,
		},
	}

	got := buildClaudeUsageFromOpenAIUsage(oai)
	require.NotNil(t, got)
	require.Equal(t, 1000, got.InputTokens, "input_tokens 必须不含 cache_read 和 cache_creation")
	require.Equal(t, 200, got.CacheReadInputTokens)
	require.Equal(t, 300, got.CacheCreationInputTokens)
	require.NotNil(t, got.CacheCreation)
	require.Equal(t, 300, got.CacheCreation.Ephemeral5mInputTokens)
	require.Equal(t, 0, got.CacheCreation.Ephemeral1hInputTokens)
}

func TestBuildClaudeUsageFromOpenAIUsage_PrefersSplit1h(t *testing.T) {
	oai := &dto.Usage{
		PromptTokens:     1500,
		CompletionTokens: 50,
		PromptTokensDetails: dto.InputTokenDetails{
			CachedTokens:         200,
			CachedCreationTokens: 300,
		},
		ClaudeCacheCreation5mTokens: 100,
		ClaudeCacheCreation1hTokens: 200,
	}

	got := buildClaudeUsageFromOpenAIUsage(oai)
	require.NotNil(t, got)
	require.Equal(t, 1000, got.InputTokens)
	require.Equal(t, 200, got.CacheReadInputTokens)
	require.Equal(t, 300, got.CacheCreationInputTokens, "aggregate cache_creation 等于 5m+1h+remainder")
	require.NotNil(t, got.CacheCreation)
	require.Equal(t, 100, got.CacheCreation.Ephemeral5mInputTokens)
	require.Equal(t, 200, got.CacheCreation.Ephemeral1hInputTokens)
}

func TestBuildClaudeUsageFromOpenAIUsage_NoCache(t *testing.T) {
	oai := &dto.Usage{
		PromptTokens:     500,
		CompletionTokens: 100,
	}

	got := buildClaudeUsageFromOpenAIUsage(oai)
	require.NotNil(t, got)
	require.Equal(t, 500, got.InputTokens)
	require.Equal(t, 100, got.OutputTokens)
	require.Equal(t, 0, got.CacheReadInputTokens)
	require.Equal(t, 0, got.CacheCreationInputTokens)
	require.Nil(t, got.CacheCreation)
}

func TestBuildClaudeUsageFromOpenAIUsage_NegativeClampedToZero(t *testing.T) {
	// 上游数据自相矛盾时不应回负数
	oai := &dto.Usage{
		PromptTokens: 100,
		PromptTokensDetails: dto.InputTokenDetails{
			CachedTokens: 200,
		},
	}

	got := buildClaudeUsageFromOpenAIUsage(oai)
	require.NotNil(t, got)
	require.Equal(t, 0, got.InputTokens)
	require.Equal(t, 200, got.CacheReadInputTokens)
}

func TestBuildClaudeUsageFromOpenAIUsage_Nil(t *testing.T) {
	require.Nil(t, buildClaudeUsageFromOpenAIUsage(nil))
}
